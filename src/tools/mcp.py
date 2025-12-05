from autogen_ext.tools.mcp import StdioServerParams

files_mcp = StdioServerParams(
    command="npx",
    args=[
        "-y",
        "@modelcontextprotocol/server-filesystem",
        r"test"
    ]
)
