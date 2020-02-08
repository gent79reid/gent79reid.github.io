---
layout: post
title: "Install Certificate on Viptela vManage"
description: ""
categories: [SD-WAN]
tags: [Cisco,Viptela,CA,Installation]
comments: true
redirect_from:
  - /2019/10/08/
---

Viptela vManage 18.4 and later offers the option to use Enterprise CA root certificate to build the certificate chain. This is a piece of good news for people who wants to build their own lab without requesting the official signed certificates from Symantec. 

Here is the high-level procedure to generate and install a signed certificate on vManage. 

**1. Install CA root on your PC**

   For MAC user, XCA is one of the user-friendly tool to manage certificates. <https://hohnstaedt.de/xca/>

   Generate self-signed root certificate which will be used to sign CSR from vManage.

   Choose "Create a self signed certificates", and use "[default] CA" template. For the first time, you will need to generate a pair of public/private key. 


   ![](/assets/img/2019-10-08/image-20191008153457839.png)

This is how it looks like after you populate other fields in the certificate form.

   ![](/assets/img/2019-10-08/image-20191008154222354.png)


**2. Install the CA root certificate on vManage**

   vManage needs to trust the root certificate before it can accepte the signed certificate from it, this is natural trustship. 

   on vManage go to "Adminstriation" -> "Controller Certificate Authorization", choose "Enterprise root certifcate"
   ![](/assets/img/2019-10-08/image-20191008154707876.png)


Paste or install CA root to it. I suggest you to export CA root certificate in .crt format and use any text editor to open up, copy and paste to here. vManage only accepts .pem format, somehow the PEM format certificates exported from XCA have some issues to get imported properly on vManage.

Next, setup CSR relevant values on the same page, 

   ![](/assets/img/2019-10-08/image-20191008155113461.png)


**3. Generate, Sign CSR and install the signed certificate**

then save and go to vManage "Configuration" -> "Certificates" -> "Controller" tab. Select vManage node and click the ... option to display the drop-down menu, generate CSR. 

   ![](/assets/img/2019-10-08/image-20191008155435365.png)

Download the CSR and import into XCA, sign it with **your CA root certificate.** 

   ![](/assets/img/2019-10-08/image-20191008155622871.png)

Export the signed CSR and install it on vManage.

   ![](/assets/img/2019-10-08/image-20191008155939110.png)

It will automatically push down to vBond, making sure you will see the successful result like this. 

   ![](/assets/img/2019-10-08/image-20191008160308281.png)

If you see any errors, double-check the root certificate which you installed on vManage is valid, and the signed certificates from CA root is also correct. 

