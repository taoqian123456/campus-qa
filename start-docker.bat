@echo off
chcp 65001 >nul
echo ============================================
echo  高校学生事务智能问答系统 - 一键启动
echo ============================================
echo.
echo [1/2] 正在构建镜像（首次约 5-10 分钟，之后秒启）...
docker compose up -d --build
if errorlevel 1 (
    echo.
    echo [失败] 构建或启动出错。请确认：
    echo   - 已安装 Docker Desktop 并已启动
    echo   - backend\.env 文件存在且填了 DEEPSEEK_API_KEY / EMBEDDING_API_KEY
    pause
    exit /b 1
)
echo.
echo [2/2] 启动完成！
echo.
echo  浏览器打开: http://localhost:8000
echo  常用命令:
echo    docker compose logs -f   查看日志
echo    docker compose down      停止（数据保留在卷里）
echo    docker compose restart   重启
echo.
pause
