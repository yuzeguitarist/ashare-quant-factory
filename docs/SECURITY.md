# SECURITY

## Secrets

- `.env` **不要提交到 git**
- systemd 环境文件建议只允许 root 读取：
  - `chmod 600 /etc/aqf/aqf.env`

## Gmail

- 强烈建议开启两步验证 + App Password
- 只授予 SMTP 所需权限，避免把主账号密码暴露在服务器上

## Data

- AQF 默认仅在本机保存 SQLite 与报告
- 若你要上传到对象存储，请自行加密并限制访问

