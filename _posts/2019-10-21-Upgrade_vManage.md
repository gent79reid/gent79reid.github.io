---
title: Upgrade Cisco vManage
date: 2019-10-21 14:10:00 +0800
categories: [SD-WAN]
tags: [Cisco,Viptela,Installation]
---

The post will walk you through the steps for upgrading vManage.

**1. Download vManage upgrade image from Cisco**

![44A898C9-580E-4F6B-8193-8B4F3E716DB1](/assets/img/2019-10-21-Upgrade_vManage/44A898C9-580E-4F6B-8193-8B4F3E716DB1.png)

The compressed file contains the new kernel, certificate, etc.

```bash
➜  vmanage-19.2.097-x86_64 ls
bootx64.efi       crash.kernel      md5sum            rootfsimg.sig     sigs.vip
cisco_crl.pem     image-signing.crt rootfs.img        sigs              vmlinuz
```

**2. Navigate to Maintenance -> Software Repository**

Add the downloaded vmanage upgrade image here. 

![2AFC4565-2CBD-4DE8-8F7E-9274C5757852](/assets/img/2019-10-21-Upgrade_vManage/2AFC4565-2CBD-4DE8-8F7E-9274C5757852.png)

After done, you should be able to see the result like below.

![FE9EA772-E83C-402E-9E4D-FA1BCA186260](/assets/img/2019-10-21-Upgrade_vManage/FE9EA772-E83C-402E-9E4D-FA1BCA186260.png)

**3. Go to Maintenance -> Software upgrade -> vManage**

Click **Upgrade** button. 

![DD05DD83-C605-4C4C-A1F0-CA9570B0C8F6](/assets/img/2019-10-21-Upgrade_vManage/DD05DD83-C605-4C4C-A1F0-CA9570B0C8F6.png)

Then select your desired version.

![DD7B3A73-F0B8-4B8C-AB82-927373A595F0](/assets/img/2019-10-21-Upgrade_vManage/DD7B3A73-F0B8-4B8C-AB82-927373A595F0.png)

Waiting for the system to install the upgrade.

![0FDFB609-739F-4D4F-836C-CEE9861163DA](/assets/img/2019-10-21-Upgrade_vManage/0FDFB609-739F-4D4F-836C-CEE9861163DA.png)

**4. Activate the newly installed upgrade version**

![95BE8602-FE90-4A17-839A-B6F061CEB66F](/assets/img/2019-10-21-Upgrade_vManage/95BE8602-FE90-4A17-839A-B6F061CEB66F.png)

This step will require a reboot, so backup your configuration and data, in case of anything goes wrong while upgrading.

![08C53C79-6219-47B5-9E06-FBC8EA100232](/assets/img/2019-10-21-Upgrade_vManage/08C53C79-6219-47B5-9E06-FBC8EA100232-1623380.png)

The last step is to set the default version. 

![B3EF6986-D775-4AAF-A68C-51F7169592DB](/assets/img/2019-10-21-Upgrade_vManage/B3EF6986-D775-4AAF-A68C-51F7169592DB.png)

![08C53C79-6219-47B5-9E06-FBC8EA100232](/assets/img/2019-10-21-Upgrade_vManage/08C53C79-6219-47B5-9E06-FBC8EA100232.png)

**Done!**

 

