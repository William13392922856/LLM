Write-Host "开始测试训练流程..."

# 运行训练器并自动输入 'y'
$process = Start-Process -FilePath ".\.venv\Scripts\python.exe" -ArgumentList "源代码\模型训练器.py" -PassThru -RedirectStandardInput "input.txt" -RedirectStandardOutput "output.txt" -RedirectStandardError "error.txt"

# 等待进程启动
Start-Sleep -Seconds 2

# 写入输入
"y" | Out-File -FilePath "input.txt" -Encoding ascii

# 等待进程完成
$process.WaitForExit()

# 读取输出
Write-Host "\n训练器输出:"
Get-Content "output.txt"

if (Test-Path "error.txt") {
    $errorContent = Get-Content "error.txt"
    if ($errorContent) {
        Write-Host "\n错误信息:"
        $errorContent
    }
}

Write-Host "\n训练器退出码: $($process.ExitCode)"

# 清理临时文件
Remove-Item "input.txt" -ErrorAction SilentlyContinue
Remove-Item "output.txt" -ErrorAction SilentlyContinue
Remove-Item "error.txt" -ErrorAction SilentlyContinue
