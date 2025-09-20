此目录用于存放应提交到仓库的测试夹具（fixtures），例如：
- 小型样例输入
- 期望对比输出（golden files）
- mock 配置等

运行时临时产物请写入 tmp_path（默认）或 TEST_OUTPUT_DIR 覆盖的目录；请勿写入本目录。