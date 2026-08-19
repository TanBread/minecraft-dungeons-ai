@echo off
cd /d "C:\Users\Nathan\Documents\AI Projects\minecraft-dungeons-ai-rs"
target\release\minecraft-dungeons-ai.exe sim --viewer --real-maps > train_log.txt 2> train_error.txt
