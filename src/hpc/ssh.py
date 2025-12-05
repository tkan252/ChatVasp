import os
import paramiko

class PasswordClusterClient:
    def __init__(
        self,
        hostname="10.251.0.28",
        username="kantang-ICME",
        password="636403f1be",
        safe_root="/home/kantang-ICME"  # 限制只能访问该目录及子目录
    ):
        self.hostname = hostname
        self.username = username
        self.password = password
        self.safe_root = os.path.normpath(safe_root)

    def _get_safe_path(self, filename: str) -> str:
        """防止路径遍历攻击（如 ../../etc/passwd）"""
        full_path = os.path.normpath(os.path.join(self.safe_root, filename))
        if not full_path.startswith(self.safe_root):
            raise PermissionError(f"Access denied: {full_path} is outside {self.safe_root}")
        return full_path

    def read_file(self, filename: str) -> str:
        """从远程读取文件"""
        remote_path = self._get_safe_path(filename)
        ssh = paramiko.SSHClient()
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        try:
            ssh.connect(
                hostname=self.hostname,
                port=22,
                username=self.username,
                password=self.password,
                timeout=10
            )
            stdin, stdout, stderr = ssh.exec_command(f"cat '{remote_path}'")
            err = stderr.read().decode().strip()
            if err:
                raise FileNotFoundError(f"Error reading file: {err}")
            return stdout.read().decode()
        finally:
            ssh.close()

    def write_file(self, filename: str, content: str) -> str:
        """向远程写入文件"""
        remote_path = self._get_safe_path(filename)
        remote_dir = os.path.dirname(remote_path)
        ssh = paramiko.SSHClient()
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        try:
            ssh.connect(
                hostname=self.hostname,
                port=22,
                username=self.username,
                password=self.password,
                timeout=10
            )
            # 创建目录
            ssh.exec_command(f"mkdir -p '{remote_dir}'")
            # 安全写入（使用 base64 避免特殊字符问题）
            import base64
            b64_content = base64.b64encode(content.encode()).decode()
            cmd = f'echo "{b64_content}" | base64 -d > "{remote_path}"'
            stdin, stdout, stderr = ssh.exec_command(cmd)
            err = stderr.read().decode().strip()
            if err:
                raise IOError(f"Write failed: {err}")
            return f"✅ Successfully wrote to {remote_path}"
        finally:
            ssh.close()

# ===== 使用示例 =====
if __name__ == "__main__":
    client = PasswordClusterClient()

    # 测试：写一个文件
    print(client.write_file("test_from_agent.txt", "Hello from Python Agent!"))

    # 测试：读回该文件
    content = client.read_file("test_from_agent.txt")
    print("Read back:", repr(content))

    # 列出家目录下的文件（可选）
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect("10.251.0.28", username="kantang-ICME", password="636403f1be")
    stdin, stdout, stderr = ssh.exec_command("ls -l /home/kantang-ICME")
    print("\nRemote files:\n", stdout.read().decode())
    ssh.close()
    