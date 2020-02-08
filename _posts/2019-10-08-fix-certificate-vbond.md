---
title: "Fix Root-ca-chain issue on Viptela vBond"
date: 2019-10-08 14:10:00 +0800
categories: [SD-WAN]
tags: [Cisco,Viptela,CA]
---

Failed to install certificate for vBond and vSmart!

![BA326A29-A9CB-4222-B4F6-020F6325EB87](/assets/img/BA326A29-A9CB-4222-B4F6-020F6325EB87.png)

I think I should install CA root certificate on vBond and vSmart first, if don’t, how the two trust the signed certificate from the CA ?
After added vBond on vManage, the CA root certificate has been pushed down it, there is no need to make manual provision. 

```bash
vBond-DC# show certificate root-ca-cert | include Issuer:
     Issuer: C=US, O=VeriSign, Inc., OU=VeriSign Trust Network, OU=(c) 2006 VeriSign, Inc. - For authorized use only, CN=VeriSign Class 3 Public Primary Certification Authority - G5
           Issuer: C=US, CN=4894d39b-4dda-4056-8f60-7671128c5a91, O=Viptela
     Issuer: C=CN, OU=R&D/emailAddress=me@gent79.me
     Issuer: C=CN, OU=R&D/emailAddress=me@gent79.me
     Issuer: C=CN, OU=R&D/emailAddress=me@gent79.me
     Issuer: C=US, O=VeriSign, Inc., OU=VeriSign Trust Network, OU=(c) 2006 VeriSign, Inc. - For authorized use only, CN=VeriSign Class 3 Public Primary Certification Authority - G5
      Issuer: C=US, O=VeriSign, Inc., OU=VeriSign Trust Network, OU=(c) 2006 VeriSign, Inc. - For authorized use only, CN=VeriSign Class 3 Public Primary Certification Authority - G5
      Issuer: C=US, O=VeriSign, Inc., OU=Class 3 Public Primary Certification Authority
      Issuer: C=US, O=DigiCert Inc, OU=www.digicert.com, CN=DigiCert Global Root G2
      Issuer: C=US, O=VeriSign, Inc., OU=VeriSign Trust Network, OU=(c) 2006 VeriSign, Inc. - For authorized use only, CN=VeriSign Class 3 Public Primary Certification Authority - G5
      Issuer: C=US, O=DigiCert Inc, OU=www.digicert.com, CN=DigiCert Global Root G2
      Issuer: C=US, O=DigiCert Inc, OU=www.digicert.com, CN=DigiCert Global Root CA
      Issuer: C=US, O=DigiCert Inc, OU=www.digicert.com, CN=DigiCert Global Root CA
```


Checking vBond root-ca-certificate, I saw 3 root certificates come from my CA, I want to remove them out and readd vBond. 

```bash
vBond-DC# request root-cert-chain uninstall 
Successfully uninstalled the root certificate chain
vBond-DC# show certificate root-ca-cert             
Error: No root-ca certificate found, or no root-ca certificate installed
```

Install root-cert manually from cli

```
​```bash
vBond-DC# request root-cert-chain install scp://root@192.168.31.79:/home/gent79/viptela-ca.pem vpn 512
Uploading root-ca-cert-chain via VPN 512
Copying ... root@192.168.31.79:/home/gent79/viptela-ca.pem via VPN 512
Warning: Permanently added '192.168.31.79' (ECDSA) to the list of known hosts.
root@192.168.31.79's password: 
viptela-ca.pem                                100% 1119   190.6KB/s   00:00    
Installing the new root certificate chain
Successfully installed the root certificate chain
vBond-DC# show certificate root-ca-cert 
Certificate:
    Data:
        Version: 3 (0x2)
        Serial Number: 7333024940781608326 (0x65c42354e45e1d86)
    Signature Algorithm: sha256WithRSAEncryption
        Issuer: C=CN, OU=R&D/emailAddress=xincheng@cisco.com
        Validity
```

Then, from vManage install the signed certificate of vBond CSR. 
![image-20191008230321074](/assets/img/image-20191008230321074.png)

```bash
vBond-DC# show certificate installed 

Server certificate
------------------

Certificate:
    Data:
        Version: 3 (0x2)
        Serial Number: 6353158359898925232 (0x582af3de2b9428b0)
    Signature Algorithm: sha256WithRSAEncryption
        Issuer: C=CN, OU=R&D/emailAddress=chengreid@gmail.com
        Validity
            Not Before: Oct  8 10:20:00 2019 GMT
            Not After : Oct  7 15:28:00 2020 GMT
        Subject: C=CN, ST=GZ, L=GY, OU=gent79, O=gent79, CN=vbond-c79b53cb-860a-4f7b-bf71-257f2f64a2de-6.vManage-DC/emailAddress=chengreid@gmail.com
        Subject Public Key Info:
            Public Key Algorithm: rsaEncryption
                Public-Key: (2048 bit)
                Modulus:
                    00:c2:2a:ac:47:86:a2:20:07:49:87:7c:21:be:53:
                    23:e9:bb:f9:d0:49:f2:f5:57:83:66:8f:93:71:84:
                    a7:d0:7d:c1:5c:f0:6c:76:6a:59:4c:f5:56:a2:e0:
                    84:94:92:9d:5a:2d:d0:be:39:4d:f6:d5:a8:aa:e4:
                    0f:38:e0:c5:9d:08:84:4f:1a:6c:36:09:c5:6a:61:
                    3e:f6:9a:94:81:d2:16:1b:0f:83:7f:73:e5:77:ba:
                    02:72:bc:b4:e1:0c:2f:ec:5b:f5:b6:fe:11:ff:2d:
                    b3:86:ef:be:86:e0:71:da:41:3a:0b:53:f3:c9:eb:
                    3e:de:1d:53:28:f2:29:67:b0:f8:b4:2d:6f:2f:10:
                    11:da:4d:99:20:07:74:7d:f7:27:fe:25:01:1a:41:
                    5b:68:1a:cf:7d:ca:b8:c7:d2:1d:f7:af:a6:cb:6a:
                    f0:8f:39:3e:be:90:4e:d2:7c:a9:4e:8a:21:cf:3b:
                    cd:e6:b0:8f:65:3b:06:ff:97:6a:fb:87:3d:2c:ec:
                    47:5e:f3:7b:7c:31:80:93:dc:33:70:78:c4:44:f2:
                    fe:12:03:75:2c:b0:43:b0:1a:60:9e:c1:00:75:1c:
                    95:59:b4:65:35:31:74:d8:12:85:35:05:53:44:b7:
                    ad:51:d8:6f:20:d4:16:fa:35:ab:3c:a2:a3:e9:a6:
                    6e:f5
                Exponent: 65537 (0x10001)
        X509v3 extensions:
            X509v3 Basic Constraints: 
                CA:FALSE
            X509v3 Subject Key Identifier: 
                C4:C2:AA:C7:2B:C1:3F:3A:B6:50:08:0E:B4:D9:5D:03:DD:C5:86:BF
    Signature Algorithm: sha256WithRSAEncryption
         18:44:9b:c9:a6:ea:06:a0:06:43:b2:43:fc:4e:b9:5f:d7:88:
         ea:6c:d4:2a:32:3c:be:a5:bb:2c:3b:04:cd:6e:6d:c4:28:0d:
         b6:2a:05:99:12:14:02:37:e8:54:29:dd:be:16:d2:d3:9f:7b:
         7b:4d:fe:6e:7b:c1:a6:bd:18:5d:64:1e:81:71:b2:6f:58:6a:
         62:9d:98:fc:d0:7f:de:42:80:19:0e:91:4f:cf:95:f7:f8:5a:
         10:33:83:cc:3b:ab:78:2d:41:8e:a3:2c:b0:4a:42:28:a5:e6:
         f8:af:9d:18:e8:eb:45:4f:65:88:39:a0:3f:83:ac:7c:c4:61:
         a3:c5:54:fb:a7:d0:09:e2:37:82:73:9a:9e:6f:5e:02:4c:4f:
         b9:73:83:d8:92:80:b1:86:60:61:5d:fb:df:7d:46:09:79:9e:
         72:c8:ea:6f:d0:1b:13:51:f1:04:7a:a4:be:6f:0b:ed:d3:8c:
         bb:06:11:83:33:ba:b1:88:71:fb:49:8a:d7:c1:fc:1c:4a:a0:
         55:7e:3a:44:8f:ad:fd:1d:4c:be:11:f9:a3:ca:e4:d8:2d:7a:
         2b:b3:60:80:b6:33:e4:9d:c5:30:25:03:28:75:0d:09:5b:4a:
         0c:fb:dc:f2:7b:74:44:05:a7:54:1e:76:58:56:38:8e:2c:e6:
         55:10:c1:72
```
