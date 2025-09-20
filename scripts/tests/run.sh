#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
BUILD_DIR="$ROOT_DIR/build"
TEST_OUTPUT_DIR_DEFAULT=""

usage() {
  echo "Usage: $0 [--persist-output] [--filter <pytest_expr>] [--gtest_filter <pattern>] [--rebuild]" >&2
  echo "  --persist-output   将 TEST_OUTPUT_DIR 指向 $ROOT_DIR/simu_sate/tests/data 以持久化产物" >&2
  echo "  --filter           传递给 pytest -k 的表达式，筛选 Python 测试" >&2
  echo "  --gtest_filter     传递给 GTest 的过滤模式（例如: *Single*）" >&2
  echo "  --rebuild          重新配置并构建 C++ 测试" >&2
}

PERSIST_OUTPUT=0
PYTEST_EXPR=
GTEST_FILTER=
REBUILD=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --persist-output)
      PERSIST_OUTPUT=1; shift ;;
    --filter)
      PYTEST_EXPR="$2"; shift 2 ;;
    --gtest_filter)
      GTEST_FILTER="$2"; shift 2 ;;
    --rebuild)
      REBUILD=1; shift ;;
    -h|--help)
      usage; exit 0 ;;
    *)
      echo "Unknown arg: $1" >&2; usage; exit 2 ;;
  esac
done

export PYTHONUNBUFFERED=1
if [[ $PERSIST_OUTPUT -eq 1 ]]; then
  export TEST_OUTPUT_DIR="$ROOT_DIR/simu_sate/tests/data"
  mkdir -p "$TEST_OUTPUT_DIR"
fi

# 1) 运行 Python 单元测试
pushd "$ROOT_DIR" >/dev/null
if [[ -n "$PYTEST_EXPR" ]]; then
  echo "[pytest] Running with -k $PYTEST_EXPR"
  pytest -q -k "$PYTEST_EXPR"
else
  pytest -q
fi
popd >/dev/null

# 2) 构建并运行 C++ GTest 单元测试
mkdir -p "$BUILD_DIR"
if [[ $REBUILD -eq 1 || ! -f "$BUILD_DIR/Makefile" && ! -f "$BUILD_DIR/build.ninja" ]]; then
  cmake -S "$ROOT_DIR/simu_sate" -B "$BUILD_DIR" -DCMAKE_BUILD_TYPE=Release
fi
cmake --build "$BUILD_DIR" --target unit_tests -j$(nproc)
ctest --test-dir "$BUILD_DIR" --output-on-failure -R unit_tests

EOF

# 生成 Linux 清理脚本
cat > scripts/tests/clean.sh <<\"EOF\"
#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
# 清理 pytest 缓存与临时输出
rm -rf "$ROOT_DIR/simu_sate/.pytest_cache" || true
# 若用户持久化了输出目录，保留 tests/data；否则清除 tmp 下由 conftest 创建的临时输出无需处理
# 可选清理：
# rm -rf "$ROOT_DIR/simu_sate/tests/data" || true
# 清理构建目录
rm -rf "$ROOT_DIR/build" || true
EOF

chmod +x scripts/tests/run.sh scripts/tests/clean.sh

# 打印结构
ls -la scripts/tests
