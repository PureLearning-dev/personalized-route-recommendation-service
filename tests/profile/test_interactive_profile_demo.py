"""路线选择反推偏好权重交互程序的端到端测试。"""

from __future__ import annotations

from pathlib import Path
import subprocess
import sys
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[2]
INTERACTIVE_SCRIPT = PROJECT_ROOT / "examples" / "interactive_profile_demo.py"


def _run_with_answers(answers: list[str]) -> subprocess.CompletedProcess[str]:
    """像真实终端一样逐行提交路线选择，并收集完整运行结果。"""

    return subprocess.run(
        [sys.executable, str(INTERACTIVE_SCRIPT)],
        input="\n".join(answers) + "\n",
        text=True,
        capture_output=True,
        cwd=PROJECT_ROOT,
        timeout=10,
        check=False,
    )


class InteractiveProfileDemoTests(unittest.TestCase):
    """验证设计好的路线属性和用户选择确实进入正式权重学习器。"""

    def test_cost_sensitive_choices_raise_cost_weight(self) -> None:
        """多次为节省费用接受其他代价时，费用应成为最高权重。"""

        result = _run_with_answers(
            [
                "b",  # 时间与费用：选择更便宜但更慢的 B
                "a",  # 时间与步行：选择更快但步行更多的 A
                "a",  # 时间与换乘：选择更快但换乘更多的 A
                "a",  # 费用与步行：选择更便宜但步行更多的 A
                "a",  # 费用与换乘：选择更便宜但换乘更多的 A
                "a",  # 步行与换乘：选择少步行但换乘更多的 A
            ]
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("已收集 6 条有效选择", result.stdout)
        self.assertIn("根据路线选择反推得到的四维偏好画像", result.stdout)
        self.assertIn("论文效用系数", result.stdout)
        self.assertIn("当前最敏感的指标：费用", result.stdout)
        self.assertIn("累计有效路线比较：6 条", result.stdout)
        self.assertIn("已选路线的平均后验概率", result.stdout)
        self.assertNotIn("长期硬约束", result.stdout)
        self.assertNotIn("本次动态画像", result.stdout)

    def test_skipped_questions_do_not_enter_learning(self) -> None:
        """跳过的题目不形成证据，全部跳过时返回等权初始结果。"""

        result = _run_with_answers(["s", "s", "s", "s", "s", "s"])

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("没有有效选择", result.stdout)
        self.assertIn("时间：25.00%", result.stdout)
        self.assertIn("费用：25.00%", result.stdout)
        self.assertIn("步行距离：25.00%", result.stdout)
        self.assertIn("换乘次数：25.00%", result.stdout)
        self.assertIn("累计有效路线比较：0 条", result.stdout)


if __name__ == "__main__":
    unittest.main()
