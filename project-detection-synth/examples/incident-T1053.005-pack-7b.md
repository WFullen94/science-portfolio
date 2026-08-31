# Detection pack — T1053.005 Scheduled Task
*Platforms: Windows · grounded on 1 MITRE analytic(s)*

> **Incident:** On host FIN-WKS-07, winword.exe spawned powershell.exe with an encoded command that queried a DNS TXT record for a randomly-named domain, then created a scheduled task named GoogleUpdateTaskMachine to persist.

## Example procedures
1. The adversary uses the `农副软件安装程序` tool to create a scheduled task named `GoogleUpdateTaskMachine` via the `schtasks` command, leveraging the credentials of the `FIN-WKS-07` host's SYSTEM account.
2. On host `FIN-WKS-07`, the actor employs PowerShell to create a scheduled task named `GoogleUpdateTaskMachine` using the `New-ScheduledTaskTrigger` and `New-ScheduledTaskAction` cmdlets, encoded within the payload of `powershell.exe`.
3. The attacker writes a custom .NET script using the `System.Scheduling.Task` library to create a persistent scheduled task named `GoogleUpdateTaskMachine` under the `SYSTEM` context, triggered to execute every 24 hours.

## Detection telemetry
- **Log/Event Sources**: Security Event Log, Task Scheduler, PowerShell
- **Key Fields**:
  - Event ID: 7045 (Scheduled Task Created)
  - Task Name: GoogleUpdateTaskMachine
  - User: SYSTEM (or another suspicious user context)
  - Command Line: `powershell.exe -EncodedCommand <base64_encoded_command>`
  - Parent Process: winword.exe
  - Child Process: powershell.exe
- **Example Command-Line**:
  - `powershell.exe -EncodedCommand <base64_encoded_command>`
- **Parent/Child Process Chains**:
  - `winword.exe -> powershell.exe`
- **Detection Analytics**:
  - **TimeWindow**: Monitor for recent activity within the last 24 hours.
  - **UserContext**: Look nå for tasks created under the SYSTEM account or other high-privilege accounts.
  - **TaskNamePattern**: Watch for any task names matching "GoogleUpdateTaskMachine" or similar patterns.
  - **CommandLineEntropyThreshold**: Analyze command lines for unusual entropy levels, indicating obfuscation or encoding.

## Sigma rule (starter)
```yaml
```yaml
title: Detection of Scheduled Task Creation by Adversary
status: experimental
logsource:
  category: process_creation
  product: windows
detection:
  selection:
    NewTaskGuid: "GoogleUpdateTaskMachine"
    ParentImage: "\Device\HarddiskVolume1\Program Files\Microsoft Office\root\Office16\WINWORD.EXE"
    Image: "\Device\HarddiskVolume1\Windows\System32\windowspowershell.v1.0\powershell.exe"
  condition: selection
falsepositives:
  - Benign process creation and task scheduling.
level: high
tags:
  - attack.execution
  - attack.t1053.005
```
```

## Test fixtures
LINE,EVENT_ID,HOST,USER,PROCESS,PARENT_PROCESS,COMMAND_LINE,NOTES
1,4697,FIN-WKS-07,SYSTEM,svchost.exe,schtasks.exe,/create /tn "GoogleUpdateTaskMachine" /tr "powershell.exe -encodedCommand ABCDEFGHIJKLMNOPQRSTUVWXYZ" ,Positive Example - Creation of scheduled task with suspicious context and command line.
2,4697,FIN-WKS-07,SYSTEM,taskeng.exe,schtasks.exe,/create /tn "GoogleUpdateTaskMachine" /tr "powershell.exe -encodedCommand ABCDEFGHIJKLMNOPQRSTUVWXYZ" ,Positive Example - Creation of scheduled task with suspicious context and commandtypedef TaskCreationEvent {
    String host;
    String user;
    String process;
    String parentProcess;
    String commandLine;
}
3,4697,FIN-WKS-07,SYSTEM,svchost.exe,schtasks.exe,/create /tn "WindowsUpdateTaskMachine" /tr "powershell.exe -encodedCommand ABCDEFGHIJKLMNOPQRSTUVWXYZ" ,Hard Negative Example - Similar command structure but legitimate task name and context.
4,4697,FIN-WKS-07,User01,explorer.exe,schtasks.exe,/create /tn "BackupTask" /tr "cmd.exe /c copy C:\Data\* E:\Backup\" ,Benign Example - Normal scheduled task creation with legitimate context and purpose.
