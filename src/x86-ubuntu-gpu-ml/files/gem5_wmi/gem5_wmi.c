#include <acpi/video.h>
#include <linux/acpi.h>
#include <linux/init.h>
#include <linux/kernel.h>
#include <linux/module.h>

MODULE_AUTHOR("Matthew Poremba");
MODULE_DESCRIPTION("gem5 fake WMI driver");
MODULE_LICENSE("Dual BSD/GPL");

acpi_status wmi_evaluate_method(const char *guid, u8 instance,
                                u32 method_id,
                                const struct acpi_buffer *in,
                                struct acpi_buffer *out)
{
    printk(KERN_ERR "gem5 fake WMI driver should not be called");
    return AE_ERROR;
}
EXPORT_SYMBOL(wmi_evaluate_method);

static int __init gem5_wmi_init(void)
{
    printk(KERN_INFO "Loading gem5 fake WMI overrides");
    return 0;
}

static void __exit gem5_wmi_exit(void)
{
    printk(KERN_INFO "Unloading gem5 fake WMI overrides");
}

module_init(gem5_wmi_init);
module_exit(gem5_wmi_exit);
