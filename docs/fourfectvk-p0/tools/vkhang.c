/* vkhang.c -- the P0 (FourFectVK T06/T08) behaviour of PanVK on kbase, through the Vulkan API.
 *
 *   vkhang hang         An endless compute dispatch (shaders/hang.comp) with a fence. Wait for it (20 s) -> expect
 *                       VK_ERROR_DEVICE_LOST when kbase hard-stops or resets; then, as an application does, free the
 *                       buffer the shader was writing at once and destroy everything. With P0 the BO must stay
 *                       mapped on the GPU until PanVK's settle proves the GPU done (kprobe kbase_gpu_munmap of our
 *                       context comes after the settle's barrier); PANVK_KBASE_P0=0 is the control (freed at once).
 *   vkhang export       The same, the submit also signalling a SYNC_FD-exportable semaphore, exported right after
 *                       the submit; poll it: when does it signal (vs the reset) and with what status.
 *   vkhang perf <n>     n times: tiny dispatch + semaphore export + poll -> submit-to-signal latency (median, p90),
 *                       to price the export barrier (PANVK_KBASE_EXPORT_BARRIER=0/1).
 *   vkhang multi        The hang with 5 more submissions queued behind it: the settle waits for all of them.
 *   vkhang watchdog     A submission that waits on a foreign fence that never signals (a raw-kbase never-ending
 *                       job ring in another kbase context, behind a FENCE_TRIGGER): vkWaitForFences(UINT64_MAX)
 *                       must return DEVICE_LOST after PANVK_KBASE_WATCHDOG_MS instead of hanging.
 * Shaders: shaders/{hang,tiny}.comp -> .inc with NDK r27 glslc -O --target-env=vulkan1.1 -mfmt=c.
 * Every step is printed with CLOCK_MONOTONIC and written to the ftrace marker (faultrun.sh: trace_clock=mono). */
#include <errno.h>
#include <fcntl.h>
#include <poll.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <time.h>
#include <unistd.h>
#include <linux/sync_file.h>
#include <vulkan/vulkan.h>

static const uint32_t hang_spv[] =
#include "shaders/hang.inc"
;
static const uint32_t tiny_spv[] =
#include "shaders/tiny.inc"
;

static int marker_fd = -1;
static double t0;
static double now_s(void)
{
   struct timespec ts;
   clock_gettime(CLOCK_MONOTONIC, &ts);
   return ts.tv_sec + ts.tv_nsec / 1e9;
}
static void say(const char *fmt, ...)
{
   char buf[512];
   va_list ap;
   va_start(ap, fmt);
   vsnprintf(buf, sizeof(buf), fmt, ap);
   va_end(ap);
   double t = now_s();
   printf("[%10.6f +%8.3f] %s\n", t, t - t0, buf);
   fflush(stdout);
   if (marker_fd >= 0)
      dprintf(marker_fd, "faultprobe: %s\n", buf);
}
#define CK(x) do { VkResult r_ = (x); if (r_ != VK_SUCCESS) { say("FAIL %s = %d", #x, r_); exit(1); } } while (0)

static int fence_status(int fd)
{
   struct sync_fence_info fi[2];
   struct sync_file_info info = {0};
   info.num_fences = 2;
   info.sync_fence_info = (uint64_t)(uintptr_t)fi;
   return ioctl(fd, SYNC_IOC_FILE_INFO, &info) ? -1000 : info.status;
}

struct vk {
   VkInstance inst;
   VkPhysicalDevice pd;
   VkDevice dev;
   VkQueue q;
   VkBuffer buf;
   VkDeviceMemory mem;
   uint32_t *map;
   VkDescriptorSetLayout dsl;
   VkPipelineLayout pl;
   VkPipeline hang, tiny;
   VkDescriptorPool dp;
   VkDescriptorSet ds;
   VkCommandPool pool;
   VkCommandBuffer cb;
   PFN_vkGetSemaphoreFdKHR get_sem_fd;
   PFN_vkImportSemaphoreFdKHR import_sem_fd;
};

static VkPipeline make_pipeline(struct vk *v, const uint32_t *code, size_t size)
{
   VkShaderModuleCreateInfo smci = { .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO, .codeSize = size,
                                     .pCode = code };
   VkShaderModule sm;
   CK(vkCreateShaderModule(v->dev, &smci, NULL, &sm));
   VkComputePipelineCreateInfo cpci = {
      .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
      .stage = { .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, .stage = VK_SHADER_STAGE_COMPUTE_BIT,
                 .module = sm, .pName = "main" },
      .layout = v->pl };
   VkPipeline p;
   CK(vkCreateComputePipelines(v->dev, VK_NULL_HANDLE, 1, &cpci, NULL, &p));
   vkDestroyShaderModule(v->dev, sm, NULL);
   return p;
}

static void vk_init(struct vk *v)
{
   memset(v, 0, sizeof(*v));
   VkApplicationInfo app = { .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO, .apiVersion = VK_API_VERSION_1_1 };
   VkInstanceCreateInfo ici = { .sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO, .pApplicationInfo = &app };
   CK(vkCreateInstance(&ici, NULL, &v->inst));
   uint32_t n = 1;
   VkResult er = vkEnumeratePhysicalDevices(v->inst, &n, &v->pd);
   if ((er != VK_SUCCESS && er != VK_INCOMPLETE) || !n) { say("FAIL no physical device"); exit(1); }
   VkPhysicalDeviceDriverProperties drv = { .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DRIVER_PROPERTIES };
   VkPhysicalDeviceProperties2 p2 = { .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2, .pNext = &drv };
   vkGetPhysicalDeviceProperties2(v->pd, &p2);
   say("device: %s | driver: %s %s", p2.properties.deviceName, drv.driverName, drv.driverInfo);
   float prio = 1.0f;
   VkDeviceQueueCreateInfo qci = { .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO, .queueFamilyIndex = 0,
                                   .queueCount = 1, .pQueuePriorities = &prio };
   const char *exts[] = { "VK_KHR_external_semaphore_fd" };
   VkDeviceCreateInfo dci = { .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO, .queueCreateInfoCount = 1,
                              .pQueueCreateInfos = &qci, .enabledExtensionCount = 1, .ppEnabledExtensionNames = exts };
   CK(vkCreateDevice(v->pd, &dci, NULL, &v->dev));
   vkGetDeviceQueue(v->dev, 0, 0, &v->q);
   v->get_sem_fd = (PFN_vkGetSemaphoreFdKHR)vkGetDeviceProcAddr(v->dev, "vkGetSemaphoreFdKHR");
   v->import_sem_fd = (PFN_vkImportSemaphoreFdKHR)vkGetDeviceProcAddr(v->dev, "vkImportSemaphoreFdKHR");

   VkBufferCreateInfo bci = { .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, .size = 65536,
                              .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT };
   CK(vkCreateBuffer(v->dev, &bci, NULL, &v->buf));
   VkMemoryRequirements mr;
   vkGetBufferMemoryRequirements(v->dev, v->buf, &mr);
   VkPhysicalDeviceMemoryProperties mp;
   vkGetPhysicalDeviceMemoryProperties(v->pd, &mp);
   uint32_t mt = UINT32_MAX;
   const VkMemoryPropertyFlags want = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
   for (uint32_t i = 0; i < mp.memoryTypeCount; i++)
      if ((mr.memoryTypeBits & (1u << i)) && (mp.memoryTypes[i].propertyFlags & want) == want) { mt = i; break; }
   VkMemoryAllocateInfo mai = { .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, .allocationSize = mr.size,
                                .memoryTypeIndex = mt };
   CK(vkAllocateMemory(v->dev, &mai, NULL, &v->mem));
   CK(vkBindBufferMemory(v->dev, v->buf, v->mem, 0));
   CK(vkMapMemory(v->dev, v->mem, 0, VK_WHOLE_SIZE, 0, (void **)&v->map));
   memset(v->map, 0, 65536);

   VkDescriptorSetLayoutBinding b = { .binding = 0, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                                      .descriptorCount = 1, .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT };
   VkDescriptorSetLayoutCreateInfo dslci = { .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
                                             .bindingCount = 1, .pBindings = &b };
   CK(vkCreateDescriptorSetLayout(v->dev, &dslci, NULL, &v->dsl));
   VkPushConstantRange pcr = { .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT, .size = 4 };
   VkPipelineLayoutCreateInfo plci = { .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO, .setLayoutCount = 1,
                                       .pSetLayouts = &v->dsl, .pushConstantRangeCount = 1,
                                       .pPushConstantRanges = &pcr };
   CK(vkCreatePipelineLayout(v->dev, &plci, NULL, &v->pl));
   v->hang = make_pipeline(v, hang_spv, sizeof(hang_spv));
   v->tiny = make_pipeline(v, tiny_spv, sizeof(tiny_spv));
   VkDescriptorPoolSize ps = { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1 };
   VkDescriptorPoolCreateInfo dpci = { .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO, .maxSets = 1,
                                       .poolSizeCount = 1, .pPoolSizes = &ps };
   CK(vkCreateDescriptorPool(v->dev, &dpci, NULL, &v->dp));
   VkDescriptorSetAllocateInfo dsai = { .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
                                        .descriptorPool = v->dp, .descriptorSetCount = 1, .pSetLayouts = &v->dsl };
   CK(vkAllocateDescriptorSets(v->dev, &dsai, &v->ds));
   VkDescriptorBufferInfo dbi = { v->buf, 0, VK_WHOLE_SIZE };
   VkWriteDescriptorSet w = { .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = v->ds, .dstBinding = 0,
                              .descriptorCount = 1, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                              .pBufferInfo = &dbi };
   vkUpdateDescriptorSets(v->dev, 1, &w, 0, NULL);
   VkCommandPoolCreateInfo pci = { .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO, .queueFamilyIndex = 0,
                                   .flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT };
   CK(vkCreateCommandPool(v->dev, &pci, NULL, &v->pool));
   VkCommandBufferAllocateInfo cai = { .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
                                       .commandPool = v->pool, .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
                                       .commandBufferCount = 1 };
   CK(vkAllocateCommandBuffers(v->dev, &cai, &v->cb));
   say("buffer mapped at %p (memory type %u)", (void *)v->map, mt);
}

static void record(struct vk *v, VkPipeline p, uint32_t value)
{
   VkCommandBufferBeginInfo bi = { .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
   CK(vkResetCommandBuffer(v->cb, 0));
   CK(vkBeginCommandBuffer(v->cb, &bi));
   vkCmdBindPipeline(v->cb, VK_PIPELINE_BIND_POINT_COMPUTE, p);
   vkCmdBindDescriptorSets(v->cb, VK_PIPELINE_BIND_POINT_COMPUTE, v->pl, 0, 1, &v->ds, 0, NULL);
   vkCmdPushConstants(v->cb, v->pl, VK_SHADER_STAGE_COMPUTE_BIT, 0, 4, &value);
   vkCmdDispatch(v->cb, 1, 1, 1);
   CK(vkEndCommandBuffer(v->cb));
}

static VkSemaphore export_sem(struct vk *v)
{
   VkExportSemaphoreCreateInfo esci = { .sType = VK_STRUCTURE_TYPE_EXPORT_SEMAPHORE_CREATE_INFO,
                                        .handleTypes = VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_SYNC_FD_BIT };
   VkSemaphoreCreateInfo sci = { .sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO, .pNext = &esci };
   VkSemaphore s;
   CK(vkCreateSemaphore(v->dev, &sci, NULL, &s));
   return s;
}

static int get_fd(struct vk *v, VkSemaphore s)
{
   VkSemaphoreGetFdInfoKHR gi = { .sType = VK_STRUCTURE_TYPE_SEMAPHORE_GET_FD_INFO_KHR, .semaphore = s,
                                  .handleType = VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_SYNC_FD_BIT };
   int fd = -1;
   VkResult r = v->get_sem_fd(v->dev, &gi, &fd);
   if (r != VK_SUCCESS)
      say("vkGetSemaphoreFdKHR = %d", r);
   return fd;
}

/* The hang, then what an application does on DEVICE_LOST: free and destroy everything. */
static int test_hang(int with_export)
{
   struct vk v;
   vk_init(&v);
   VkFence f;
   VkFenceCreateInfo fci = { .sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO };
   CK(vkCreateFence(v.dev, &fci, NULL, &f));
   VkSemaphore sem = with_export ? export_sem(&v) : VK_NULL_HANDLE;
   record(&v, v.hang, 0);
   VkSubmitInfo si = { .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO, .commandBufferCount = 1, .pCommandBuffers = &v.cb,
                       .signalSemaphoreCount = with_export ? 1 : 0, .pSignalSemaphores = &sem };
   double ts = now_s();
   CK(vkQueueSubmit(v.q, 1, &si, f));
   say("hang dispatch submitted%s", with_export ? " (signals an exportable semaphore)" : "");
   int sfd = with_export ? get_fd(&v, sem) : -1;
   if (with_export)
      say("exported SYNC_FD %d, status %d", sfd, sfd >= 0 ? fence_status(sfd) : -1);

   VkResult wr = VK_TIMEOUT;
   double tfence = 0;
   if (sfd >= 0) {
      /* Poll the exported fence; check the Vulkan fence each time around. */
      while (now_s() - ts < 20) {
         struct pollfd p = { sfd, POLLIN, 0 };
         if (poll(&p, 1, 50) > 0 && !tfence) {
            tfence = now_s();
            say("EXPORTED FENCE signalled, status %d, %.3f s after submit", fence_status(sfd), tfence - ts);
         }
         wr = vkWaitForFences(v.dev, 1, &f, VK_TRUE, 0);
         if (wr != VK_TIMEOUT && tfence)
            break;
      }
   } else {
      wr = vkWaitForFences(v.dev, 1, &f, VK_TRUE, 20000000000ull);
   }
   say("vkWaitForFences = %d (%s) after %.3f s; counter as the CPU sees it %u", wr,
       wr == VK_ERROR_DEVICE_LOST ? "DEVICE_LOST" : wr == VK_SUCCESS ? "SUCCESS" : "?", now_s() - ts, v.map[0]);

   /* The application frees right away. */
   say("APP FREES the buffer memory");
   vkUnmapMemory(v.dev, v.mem);
   vkFreeMemory(v.dev, v.mem, NULL);
   vkDestroyBuffer(v.dev, v.buf, NULL);
   say("APP FREED the buffer memory");
   vkFreeCommandBuffers(v.dev, v.pool, 1, &v.cb);
   vkDestroyCommandPool(v.dev, v.pool, NULL);
   vkDestroyPipeline(v.dev, v.hang, NULL);
   vkDestroyPipeline(v.dev, v.tiny, NULL);
   vkDestroyDescriptorPool(v.dev, v.dp, NULL);
   vkDestroyPipelineLayout(v.dev, v.pl, NULL);
   vkDestroyDescriptorSetLayout(v.dev, v.dsl, NULL);
   vkDestroyFence(v.dev, f, NULL);
   if (with_export)
      vkDestroySemaphore(v.dev, sem, NULL);
   say("app objects destroyed; waiting 3 s before vkDestroyDevice");
   usleep(3000000);
   double td = now_s();
   vkDestroyDevice(v.dev, NULL);
   say("vkDestroyDevice took %.3f ms", (now_s() - td) * 1e3);
   vkDestroyInstance(v.inst, NULL);
   if (sfd >= 0)
      close(sfd);
   say("RESULT %s", wr == VK_ERROR_DEVICE_LOST ? "DEVICE_LOST as expected" : "UNEXPECTED");
   return wr == VK_ERROR_DEVICE_LOST ? 0 : 1;
}

static int cmp_d(const void *a, const void *b)
{
   double x = *(const double *)a, y = *(const double *)b;
   return x < y ? -1 : x > y;
}

static int test_perf(int n)
{
   struct vk v;
   vk_init(&v);
   double *lat = calloc(n, sizeof(double));
   VkSemaphore sem = export_sem(&v);
   int bad = 0;
   for (int i = 0; i < n; i++) {
      record(&v, v.tiny, 0x1000 + i);
      VkSubmitInfo si = { .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO, .commandBufferCount = 1,
                          .pCommandBuffers = &v.cb, .signalSemaphoreCount = 1, .pSignalSemaphores = &sem };
      double ts = now_s();
      CK(vkQueueSubmit(v.q, 1, &si, VK_NULL_HANDLE));
      int fd = get_fd(&v, sem); /* SYNC_FD export resets the semaphore (copy transference) */
      if (fd >= 0) {
         struct pollfd p = { fd, POLLIN, 0 };
         poll(&p, 1, 5000);
         lat[i] = (now_s() - ts) * 1e6;
         bad += fence_status(fd) != 1;
         close(fd);
      } else {
         lat[i] = (now_s() - ts) * 1e6; /* already complete: no fence to wait for */
      }
      CK(vkQueueWaitIdle(v.q));
      bad += v.map[1] != 0x1000u + i;
   }
   qsort(lat, n, sizeof(double), cmp_d);
   say("perf: %d submits with SYNC_FD export: submit->signal median %.0f us, p10 %.0f, p90 %.0f; %d bad", n,
       lat[n / 2], lat[n / 10], lat[n * 9 / 10], bad);
   vkDestroyDevice(v.dev, NULL);
   return bad != 0;
}

/* ---- a foreign fence that never signals: a raw kbase context running a never-ending, soft-stoppable
 * job ring (two WRITE_VALUE jobs clearing each other's status, Suppress Prefetch) behind a FENCE_TRIGGER. */
#define KB_TYPE 0x80
struct kb_ver { uint16_t major, minor; };
struct kb_flags { uint32_t f; };
struct kb_submit { uint64_t addr; uint32_t nr, stride; };
union kb_alloc { struct { uint64_t va_pages, commit_pages, extension, flags; } in; struct { uint64_t flags, gpu_va; } out; };
struct kb_atom { uint64_t jc, udata[2], extres; uint16_t nr_extres; uint8_t jit[2]; uint8_t dep[4];
                 uint8_t num, prio, devnr, slot; uint32_t req; uint8_t pad[8]; uint32_t frame, pad2; };

static int never_signalled_fence(void)
{
   int fd = open("/dev/mali0", O_RDWR | O_CLOEXEC);
   struct kb_ver ver = {0, 0};
   struct kb_flags fl = {0};
   ioctl(fd, _IOWR(KB_TYPE, 0, struct kb_ver), &ver);
   ioctl(fd, _IOW(KB_TYPE, 1, struct kb_flags), &fl);
   mmap(NULL, 4096, PROT_NONE, MAP_SHARED, fd, 3ull << 12);
   union kb_alloc a = { .in = { 1, 1, 0, 0xf } };
   if (ioctl(fd, _IOWR(KB_TYPE, 5, union kb_alloc), &a)) { say("raw MEM_ALLOC failed"); return -1; }
   uint32_t *d = mmap(NULL, 4096, PROT_READ | PROT_WRITE, MAP_SHARED, fd, (off_t)a.out.gpu_va);
   memset(d, 0, 4096);
   const uint64_t j0 = (uint64_t)(uintptr_t)d, j1 = j0 + 128;
   uint32_t *h[2] = { d, d + 32 };
   for (int i = 0; i < 2; i++) {
      const uint64_t other = i ? j0 : j1;
      h[i][4] = (2u << 1) | (1u << 11) | ((i + 1u) << 16); /* WRITE_VALUE, suppress prefetch, index */
      h[i][6] = (uint32_t)other; h[i][7] = (uint32_t)(other >> 32);       /* next = the other */
      h[i][8] = (uint32_t)other; h[i][9] = (uint32_t)(other >> 32);       /* write to the other's status */
      h[i][10] = 7;                                                        /* immediate 64: 0 */
   }
   __sync_synchronize();
   struct { int fd, stream_fd; } bf = { -1, -1 };
   struct kb_atom at[2];
   memset(at, 0, sizeof(at));
   at[0].jc = j0; at[0].num = 1; at[0].req = 1u << 1;
   at[1].jc = (uint64_t)(uintptr_t)&bf; at[1].num = 2; at[1].req = (1u << 9) | 0x2; at[1].dep[0] = 1; at[1].dep[1] = 1;
   struct kb_submit s = { (uint64_t)(uintptr_t)at, 2, 64 };
   if (ioctl(fd, _IOW(KB_TYPE, 2, struct kb_submit), &s)) { say("raw JOB_SUBMIT failed"); return -1; }
   say("never-signalled foreign fence: fd %d from a raw kbase context (never-ending ring)", bf.fd);
   return bf.fd;
}

static int test_watchdog(void)
{
   int ffd = never_signalled_fence();
   if (ffd < 0)
      return 1;
   struct vk v;
   vk_init(&v);
   VkSemaphoreCreateInfo sci = { .sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO };
   VkSemaphore wsem;
   CK(vkCreateSemaphore(v.dev, &sci, NULL, &wsem));
   VkImportSemaphoreFdInfoKHR ii = { .sType = VK_STRUCTURE_TYPE_IMPORT_SEMAPHORE_FD_INFO_KHR, .semaphore = wsem,
                                     .flags = VK_SEMAPHORE_IMPORT_TEMPORARY_BIT,
                                     .handleType = VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_SYNC_FD_BIT, .fd = ffd };
   CK(v.import_sem_fd(v.dev, &ii));
   VkFence f;
   VkFenceCreateInfo fci = { .sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO };
   CK(vkCreateFence(v.dev, &fci, NULL, &f));
   record(&v, v.tiny, 0x77);
   VkPipelineStageFlags st = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
   VkSubmitInfo si = { .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO, .waitSemaphoreCount = 1, .pWaitSemaphores = &wsem,
                       .pWaitDstStageMask = &st, .commandBufferCount = 1, .pCommandBuffers = &v.cb };
   double ts = now_s();
   CK(vkQueueSubmit(v.q, 1, &si, f));
   say("submitted work waiting on the never-signalled fence; vkWaitForFences(UINT64_MAX)");
   VkResult wr = vkWaitForFences(v.dev, 1, &f, VK_TRUE, UINT64_MAX);
   say("vkWaitForFences = %d (%s) after %.3f s; v[1] = 0x%x", wr, wr == VK_ERROR_DEVICE_LOST ? "DEVICE_LOST" : "?",
       now_s() - ts, v.map[1]);
   say("APP FREES the buffer memory");
   vkUnmapMemory(v.dev, v.mem);
   vkFreeMemory(v.dev, v.mem, NULL);
   say("APP FREED the buffer memory");
   double td = now_s();
   vkDestroyDevice(v.dev, NULL);
   say("vkDestroyDevice took %.3f ms", (now_s() - td) * 1e3);
   say("RESULT %s", wr == VK_ERROR_DEVICE_LOST ? "DEVICE_LOST by the watchdog" : "UNEXPECTED");
   return wr == VK_ERROR_DEVICE_LOST ? 0 : 1;
}

/* The hang with 5 more submissions queued behind it (each ORDER-dependent on the previous in PanVK's queue):
 * they run after the hang ends; the settle must wait for all of them before its barrier. */
static int test_multi(void)
{
   struct vk v;
   vk_init(&v);
   VkCommandBuffer cbs[6];
   VkCommandBufferAllocateInfo cai = { .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
                                       .commandPool = v.pool, .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
                                       .commandBufferCount = 6 };
   CK(vkAllocateCommandBuffers(v.dev, &cai, cbs));
   for (int i = 0; i < 6; i++) {
      v.cb = cbs[i];
      record(&v, i ? v.tiny : v.hang, 0x100 + i);
   }
   VkFence f;
   VkFenceCreateInfo fci = { .sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO };
   CK(vkCreateFence(v.dev, &fci, NULL, &f));
   double ts = now_s();
   for (int i = 0; i < 6; i++) {
      VkSubmitInfo si = { .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO, .commandBufferCount = 1, .pCommandBuffers = &cbs[i] };
      CK(vkQueueSubmit(v.q, 1, &si, i == 5 ? f : VK_NULL_HANDLE));
   }
   say("hang + 5 tiny submissions queued; waiting on the last one's fence");
   VkResult wr = vkWaitForFences(v.dev, 1, &f, VK_TRUE, 20000000000ull);
   say("vkWaitForFences = %d (%s) after %.3f s; v[1] = 0x%x", wr, wr == VK_ERROR_DEVICE_LOST ? "DEVICE_LOST" : "?",
       now_s() - ts, v.map[1]);
   say("APP FREES the buffer memory");
   vkUnmapMemory(v.dev, v.mem);
   vkFreeMemory(v.dev, v.mem, NULL);
   say("APP FREED the buffer memory");
   usleep(3000000);
   double td = now_s();
   vkDestroyDevice(v.dev, NULL);
   say("vkDestroyDevice took %.3f ms", (now_s() - td) * 1e3);
   say("RESULT %s", wr == VK_ERROR_DEVICE_LOST ? "DEVICE_LOST as expected" : "UNEXPECTED");
   return wr == VK_ERROR_DEVICE_LOST ? 0 : 1;
}

int main(int argc, char **argv)
{
   setvbuf(stdout, NULL, _IOLBF, 0);
   t0 = now_s();
   marker_fd = open("/sys/kernel/tracing/trace_marker", O_WRONLY | O_CLOEXEC);
   const char *t = argc > 1 ? argv[1] : "";
   say("vkhang %s (PANVK_KBASE_P0=%s EXPORT_BARRIER=%s WATCHDOG_MS=%s)", t, getenv("PANVK_KBASE_P0") ?: "-",
       getenv("PANVK_KBASE_EXPORT_BARRIER") ?: "-", getenv("PANVK_KBASE_WATCHDOG_MS") ?: "-");
   if (!strcmp(t, "hang"))
      return test_hang(0);
   if (!strcmp(t, "export"))
      return test_hang(1);
   if (!strcmp(t, "perf"))
      return test_perf(argc > 2 ? atoi(argv[2]) : 200);
   if (!strcmp(t, "multi"))
      return test_multi();
   if (!strcmp(t, "watchdog"))
      return test_watchdog();
   printf("usage: vkhang hang | export | multi | perf <n> | watchdog\n");
   return 2;
}
