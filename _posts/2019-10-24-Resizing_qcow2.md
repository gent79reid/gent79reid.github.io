---
layout: post
title: "Resizing CSR1kV Qcow2 Image"
description: "This post will show you the simple steps of resizing CSR1kV image"
categories: [SD-WAN]
tags: [Cisco,Qemu,Installation]
comments: true
---


The size of CSR1000v qcow2 image downloaded from Cisco is only 8G, which is below the minimum requirement of SD-AVC Controller installation. And I don't want to reinstall CSR1kV from ISO, let's try to resize the image. 

**Step 1: Identify the filesytem used on CSR1kV image**

```bash
root@eve-ng:# virt-filesystems -a virtioa.qcow2 -l
Name       Type        VFS   Label      Size         Parent
/dev/sda1  filesystem  ext2  bootflash  8433200128  -
/dev/sda5  filesystem  ext2  bootflash  536870400    -
```

**Step 2: Resize the image**

```bash
root@eve-ng:# qemu-img resize virtoa.qcow2 +20G

root@eve-ng:# virt-resize --expand /dev/sda1 virtioa.qcow2 virtioa-new.qcow2 
[   0.0] Examining virtioa.qcow2
 100% ⟦▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒⟧ --:--

**********

Summary of changes:

/dev/sda1: This partition will be resized from 7.4G to 27.4G.  The 
filesystem ext2 on /dev/sda1 will be expanded using the 'resize2fs' method.

/dev/sda2: This partition will be left alone.

/dev/sda3: This partition will be left alone.

/dev/sda4: This partition will be left alone.

**********

[   7.9] Setting up initial partition table on virtioa-1.qcow2
[   8.9] Copying /dev/sda1
◓ 78% ⟦▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒═════════════════════════════════⟧ 00:03
```

**Step 3: Replace the original image**

```bash
root@eve-ng:#mv virtioa-new.qcow2 virtioa.qcow2
```



**Note** If you are using eve-ng, you have to remove the CSR1kV node and add it back. 


