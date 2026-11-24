@echo off
chcp 65001 >nul
title Magna SE Mini - PCB & Circuit Studio
python "%~dp0magna_studio.py"
if errorlevel 1 pause
