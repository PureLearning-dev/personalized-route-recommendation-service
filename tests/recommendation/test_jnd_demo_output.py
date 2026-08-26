"""验证JND演示能够完整展示路线选择、候选路线和比较过程。"""

from __future__ import annotations

from contextlib import redirect_stdout
from io import StringIO
import unittest

from examples.jnd_enhanced_route_ranking_demo import main


class JndDemoOutputTests(unittest.TestCase):
    def test_demo_prints_complete_explainable_process(self) -> None:
        output = StringIO()
        with redirect_stdout(output):
            main()

        text = output.getvalue()
        expected_fragments = (
            "步骤1：展示代码中预设的路线选择",
            "第1题：时间与费用",
            "代码预设选择：A",
            "累计6次选择后的画像",
            "步骤2：展示本次推荐的候选路线和硬约束过滤",
            "too-much-walking",
            "被过滤：超过步行距离上限",
            "步骤3：展示每条可行路线的归一化、分项贡献和加权初排",
            "原始值 40 → 归一化",
            "加权代价越小，初排越靠前",
            "步骤4：展示JND逐层比较的中间过程",
            "第1轮：比较 slow-cheap、balanced、fast-expensive",
            "差异不明显，进入下一指标继续比较",
            "差异已经明显，可由当前指标区分",
            "步骤5：输出最终Top-K路线，供用户选择",
            "最终推荐Top-2如下",
            "推荐路线1：balanced",
            "推荐路线2：slow-cheap",
            "用户从Top-2中选择：推荐路线1（balanced）",
        )
        for fragment in expected_fragments:
            with self.subTest(fragment=fragment):
                self.assertIn(fragment, text)


if __name__ == "__main__":
    unittest.main()
