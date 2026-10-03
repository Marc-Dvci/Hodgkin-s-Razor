"""Write film/story.json: the film's beats, narration, Chinese subtitles and scenes.

    python film/build_story.py
    python film/narrate.py
    python film/record.py

Every image is a capture of the running application (results/capture, made by
scripts/capture_app.py) or a figure written from the results. The terminal
scene is the real output of `python demo.py`, captured when this script runs.
narrate.py refuses any spoken number that the results files do not contain.
"""
from __future__ import annotations

import html
import json
import pathlib
import subprocess
import sys

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent
CAP = "../results/capture/"
V3 = "../results/v3/figures/"
V2 = "../results/v2/figures/"
SITE = "../site/index.html"


def demo_output() -> str:
    out = subprocess.run([sys.executable, str(ROOT / "demo.py")], capture_output=True,
                         text=True, cwd=ROOT, check=True).stdout
    lines = [l.rstrip() for l in out.splitlines()]
    start = next(i for i, l in enumerate(lines) if l.startswith("0. Blind test"))
    end = next(i for i, l in enumerate(lines) if l.startswith("1. Blind test"))
    head = ["$ python demo.py", ""] + lines[start:end - 1]
    return html.escape("\n".join(head))


def beats() -> list[dict]:
    return [
        {"scene": {"type": "title", "kicker": "AI4S Open Innovation · End-to-End System",
                   "title": "Hodgkin's Razor",
                   "lead": "A mechanistic digital twin for neural organ-on-chip recordings"},
         "say": "A microelectrode array under a neural organ-on-chip records every burst of a "
                "living network. Then the analysis ends in a description: firing fell, bursts "
                "shortened.",
         "zh": "神经器官芯片下的微电极阵列记录下活体网络的每一次爆发。但分析最终只停留在描述：放电减少，爆发变短。"},
        {"scene": {"type": "image", "kicker": "The problem", "title": "What did the compound do?",
                   "src": CAP + "13_ttx_raster.png", "dark": True},
         "say": "It does not say what the compound did. Blocking AMPA receptors, opening chloride "
                "channels or shutting sodium channels can all silence a network the same way.",
         "zh": "它无法说明化合物做了什么。阻断 AMPA 受体、打开氯离子通道或关闭钠通道，都可能以同样的方式让网络沉寂。"},
        {"scene": {"type": "steps", "kicker": "The system", "title": "From a recording to a mechanism",
                   "items": ["Two recordings: baseline and treated",
                             "The array's own electrodes and dead time, applied to every simulation",
                             "A GPU-simulated biophysical network, fitted to the recording",
                             "Probability, size and interval for ten mechanisms",
                             "A guard that refuses when the twin cannot reproduce the data"]},
         "say": "Hodgkin's Razor fits a biophysical network, simulated on a GPU, to the recording "
                "itself. It is trained only on simulations, and has never seen a compound label.",
         "zh": "Hodgkin's Razor 把在 GPU 上模拟的生物物理神经网络直接拟合到记录本身。它只用模拟数据训练，从未见过任何化合物标签。"},
        {"scene": {"type": "live", "kicker": "The running application · rat cortical culture, after TTX",
                   "src": SITE + "#analyse"},
         "actions": [{"cue": 0, "delay": 200, "do": "select", "sel": "#example", "value": "rat_ttx"},
                     {"cue": 0, "delay": 1400, "do": "click", "sel": "#run"},
                     {"cue": 0, "delay": 2600, "do": "scroll", "sel": "#mechpanel"},
                     {"cue": 1, "delay": 400, "do": "point", "sel": "#mech tbody tr"}],
         "say": "For ten mechanisms it returns the probability that each one moved, the size of the "
                "shift, and an interval. Here, a rat culture after TTX: sodium channels, down, "
                "probability 0.65.",
         "zh": "对十种机制，它给出每种机制发生变化的概率、变化幅度和区间。这里是施加 TTX 后的大鼠培养物：钠通道下降，概率 0.65。"},
        {"scene": {"type": "live", "continue": True,
                   "kicker": "Not a lookup · measured, against the twin re-simulated"},
         "actions": [{"cue": 0, "delay": 0, "do": "scroll", "sel": "#rasterpanel"}],
         "say": "Before it names anything, it re-simulates the network at the parameters it inferred, "
                "and checks that the twin reproduces what was measured.",
         "zh": "在给出任何结论之前，它会在推断出的参数下重新模拟网络，检查孪生模型能否再现测量结果。"},
        {"scene": {"type": "live", "continue": True,
                   "kicker": "The refusal · human culture, outside the model"},
         "actions": [{"cue": 0, "delay": 0, "do": "scroll", "sel": "body", "top": True},
                     {"cue": 0, "delay": 500, "do": "select", "sel": "#example", "value": "human_gaba"},
                     {"cue": 0, "delay": 1300, "do": "click", "sel": "#run"},
                     {"cue": 1, "delay": 1000, "do": "point", "sel": "#verdict"}],
         "say": "When it cannot, it says so. This human culture is outside the model, and no "
                "mechanism is named.",
         "zh": "如果不能再现，它会明确说出来。这个人源培养物超出了模型范围，因此不给出任何机制。"},
        {"scene": {"type": "image", "kicker": "The guard, tested", "title": "Cases it must refuse, cases it must pass",
                   "src": V2 + "guard.png"},
         "say": "The refusal is tested. On one recording system it fired on 0.95 of simulated cultures "
                "with receptor kinetics it cannot represent, on every shuffled recording, and on only "
                "0.02 of held-out simulations. On the other two systems it is weaker, and the report "
                "says so.",
         "zh": "拒绝机制经过了检验。在一种记录系统上，它对 0.95 的含有无法表示的受体动力学的模拟培养物、对所有被打乱的记录都会触发，而对留出的模拟数据只有 0.02 触发。在另外两种系统上它较弱，报告中如实说明。"},
        {"scene": {"type": "quote", "kicker": "The rule behind every result",
                   "text": "A method's chance level is its own hit rate when nothing was applied.",
                   "source": "Every test is scored against untreated recordings read the same way."},
         "say": "Every result is scored against untreated recordings, read in exactly the same way. "
                "A method's chance level is its own hit rate when nothing was applied.",
         "zh": "每一项结果都与以完全相同方式读取的未处理记录对照评分。一种方法的机会水平，就是在未施药时它自身的命中率。"},
        {"scene": {"type": "numbers", "kicker": "What the rule caught", "title": "A comparator on a blind test",
                   "items": [{"value": "8", "of": "/10", "cap": "treated wells, named the accepted mechanism"},
                             {"value": "6", "of": "/10", "cap": "untreated pairs, named the same mechanisms"}],
                   "pills": [["no", "a preference, not a detection"]]},
         "say": "That rule caught a comparator. It scored 8 of 10 on a blind test, and named the same "
                "mechanisms on 6 of 10 untreated pairs. A preference, not a detection.",
         "zh": "这条规则识破了一个对照方法。它在盲测中得到 10 中 8，但在 10 对未处理样本中有 6 对给出同样的机制。这是一种偏好，而不是检测。"},
        {"scene": {"type": "title", "kicker": "Pre-registered blind test",
                   "title": "Chronic NMDA blockade on sister cultures",
                   "lead": "Charlesworth et al. 2015 · one sister of each preparation kept on APV for days",
                   "hash": "PREREGISTRATION_v3 · sha256 2e1e630e…"},
         "say": "The blind test. Charlesworth and colleagues kept one sister culture of each "
                "preparation on an NMDA blocker for days. The pre-registration was hashed before a "
                "single treated recording was read.",
         "zh": "盲测。Charlesworth 等人让每份制备中的一个姊妹培养物持续数天处于 NMDA 阻断剂中。在读取任何一个处理后的记录之前，预注册就已经生成哈希并锁定。"},
        {"scene": {"type": "live", "kicker": "Result · the blind test, in the application",
                   "src": SITE + "#blind"},
         "actions": [{"cue": 0, "delay": 600, "do": "point", "sel": "#v3kpis .kpi"},
                     {"cue": 1, "delay": 300, "do": "scroll", "sel": "#v3method", "offset": 70},
                     {"cue": 2, "delay": 0, "do": "click", "sel": "#v3method button[data-m=unpaired]"},
                     {"cue": 3, "delay": 0, "do": "click", "sel": "#v3method button[data-m=prior_art]"}],
         "say": "The twin's NMDA reading separates 29 treated preparations from 23 untreated sisters "
                "of the same genotypes. AUROC 0.86, against a pre-registered bar of 0.70. "
                "Remove the pairing, and it falls to 0.66. The published estimator it builds on "
                "scores 0.49 on the same cultures.",
         "zh": "孪生模型的 NMDA 读数把 29 份处理过的制备与 23 份同基因型的未处理姊妹培养物区分开：AUROC 0.86，预设门槛 0.70。去掉配对后降到 0.66。它所基于的已发表估计器在同一批培养物上只得 0.49。"},
        {"scene": {"type": "live", "continue": True, "kicker": "Result · ten mechanisms, no drug label"},
         "actions": [{"cue": 0, "delay": 0, "do": "click", "sel": "#v3method button[data-m=twin]"},
                     {"cue": 1, "delay": 200, "do": "click", "sel": "#v3profile g.hit:nth-of-type(2)"},
                     {"cue": 1, "delay": 2600, "do": "click", "sel": "#v3profile g.hit:nth-of-type(1)"}],
         "say": "It was never told which drug was applied. Of all ten mechanisms it reads, "
                "NMDA is the one that separates treated from untreated best.",
         "zh": "它从未被告知施加了哪种药物。在它读取的全部十种机制中，NMDA 是区分处理组与未处理组最好的一个。"},
        {"scene": {"type": "live", "continue": True,
                   "kicker": "Result · the reading fades as the cultures compensate"},
         "actions": [{"cue": 0, "delay": 0, "do": "scroll", "sel": "#v3canal", "offset": 60}],
         "say": "The cultures compensate as they mature, as the authors reported. The twin's reading "
                "fades with them.",
         "zh": "正如原作者所报告的，培养物在成熟过程中发生代偿。孪生模型的读数也随之减弱。"},
        {"scene": {"type": "live", "continue": True, "kicker": "What failed, reported with the same weight"},
         "actions": [{"cue": 0, "delay": 0, "do": "scroll", "sel": "#v3fail", "offset": 160},
                     {"cue": 2, "delay": 0, "do": "scroll", "sel": "#dyn", "offset": 60},
                     {"cue": 2, "delay": 1500, "do": "point", "sel": "#dynkpis .kpi:nth-child(3)"},
                     {"cue": 2, "delay": 4200, "do": "point", "sel": "#dynkpis .kpi:nth-child(1)"}],
         "say": "What failed is reported with the same weight. Only 2 of 29 treated cultures had "
                "NMDA as the single top mechanism. On an earlier Dynasore test, the exact mechanism was "
                "right in 2 of 10 wells, though the drug was detected in 9 of 10, and in none of the "
                "untreated pairs.",
         "zh": "失败的结果同样如实报告。只有 29 份处理过的培养物中的 2 份把 NMDA 列为首位机制。在此前的 Dynasore 测试中，精确机制只在 10 个孔中对了 2 个，但药物效应在 10 个孔中检出 9 个，在未处理样本中一个也没有误报。"},
        {"scene": {"type": "live", "kicker": "The chip twin · chip planner, in the application",
                   "src": SITE + "#chips"},
         "actions": [{"cue": 1, "delay": 0, "do": "point", "sel": "#heat tbody tr:nth-child(3) td:nth-child(3)"}],
         "say": "The same simulator runs a two-compartment chip with one-way microchannels, the "
                "geometry of the sponsor's own devices. It says, before the experiment, which readout "
                "resolves which property.",
         "zh": "同一个模拟器还能运行带单向微通道的双腔芯片，这正是赞助方自己器件的结构。它在实验之前就能指出，哪种读数可以分辨哪种性质。"},
        {"scene": {"type": "live", "continue": True, "kicker": "The chip twin · how many chips a claim needs"},
         "actions": [{"cue": 0, "delay": 0, "do": "scroll", "sel": "#chipslider", "offset": 90},
                     {"cue": 0, "delay": 900, "do": "slide", "sel": "#chipslider", "from": 2, "to": 6}],
         "say": "The one public test of chip directionality used about 8 chips per design. The twin "
                "says it needed 20.",
         "zh": "唯一公开的芯片方向性测试每种设计只用了约 8 块芯片。孪生模型指出需要 20 块。"},
        {"scene": {"type": "title", "kicker": "The sponsor laboratory's cortico-striatal chip",
                   "title": "One failed prediction, one revision, both committed before running",
                   "pills": [["ok", "striato-striatal synchrony lower"], ["ok", "cortico-striatal synchrony lower"],
                             ["no", "event frequency not reproduced"]]},
         "say": "On the sponsor laboratory's cortico-striatal result, the first committed prediction "
                "failed. The revision imposed only the paper's own finding, that an isolated striatum "
                "is silent. Both synchrony effects then appeared, in the published direction. Event "
                "frequency did not.",
         "zh": "对于赞助方实验室的皮层-纹状体结果，第一次提交的预测失败了。修订版只加入了论文本身的发现：孤立的纹状体是沉默的。随后两种同步性效应都按已发表的方向出现，但事件频率没有。"},
        {"scene": {"type": "quote", "kicker": "Why it matters",
                   "text": "From experimental description toward predictive simulation.",
                   "source": "The supporting organisation's stated aim for neural organ-on-chip data"},
         "say": "For a neural organ-on-chip laboratory, a recording becomes a mechanism hypothesis, "
                "with the untreated comparison that says how far to trust it. And before the "
                "experiment, the twin says which electrodes to use and how many chips to run. That "
                "is the step from describing an experiment to predicting one.",
         "zh": "对于神经器官芯片实验室来说，一次记录变成了一个机制假设，并附带说明可信程度的未处理对照。在实验之前，孪生模型还会指出该用哪些电极、需要多少块芯片。这就是从描述实验走向预测实验的一步。"},
        {"scene": {"type": "terminal", "kicker": "Run it", "title": "Public data, one desktop GPU, the demo on a CPU",
                   "text": demo_output()},
         "say": "Everything runs from public data on one desktop GPU. The demo runs on a CPU, and one "
                "script checks every hash, every frozen model, and every written number.",
         "zh": "一切都基于公开数据，在一块桌面 GPU 上运行。演示可在 CPU 上运行，一个脚本会核对每一个哈希、每一个冻结的模型和每一个写出的数字。"},
        {"scene": {"type": "title", "kicker": "Hodgkin's Razor",
                   "title": "From describing an experiment, to predicting it, and knowing when not to.",
                   "lead": "github.com/Marc-Dvci/Hodgkin-s-Razor"},
         "say": "From describing an experiment, to predicting it. And knowing when not to.",
         "zh": "从描述实验，到预测实验，并且知道何时不该预测。", "hold_ms": 1500},
    ]


def script_md(bs: list[dict]) -> str:
    """docs/VIDEO_SCRIPT.md, from the same beats, with measured times if narrated."""
    tp = HERE / "timing.json"
    dur = json.loads(tp.read_text())["durations"] if tp.exists() else None
    if dur and len(dur) != len(bs):
        dur = None
    out = ["# Demo video script", "",
           "Generated by `python film/build_story.py` from the same beats the film is rendered "
           "from. Times are measured from the synthesised narration (`film/timing.json`). "
           "Every image is a capture of the running application or a figure written from the "
           "results; `film/narrate.py` refuses any spoken number the results files do not "
           "contain.", ""]
    t = 0
    for i, b in enumerate(bs):
        sc = b["scene"]
        head = f"## {i + 1}. {sc.get('kicker', '')}: {sc.get('title', '')}".rstrip(": ")
        if dur:
            head += f" ({t // 60000}:{(t // 1000) % 60:02d})"
            t += dur[i]
        out += [head, "", f"**Screen.** {sc['type']}"
                + (f", `{sc['src'].replace('../', '')}`" if sc.get("src") else ""), "",
                f"**Narration.** {b['say']}", "", f"**中文字幕.** {b['zh']}", ""]
    if dur:
        out.append(f"Total: {sum(dur) / 1000:.1f} s.")
    return "\n".join(out) + "\n"


def main() -> None:
    bs = beats()
    (HERE / "story.json").write_text(json.dumps({"beats": bs}, ensure_ascii=False, indent=1),
                                     encoding="utf-8")
    (ROOT / "docs" / "VIDEO_SCRIPT.md").write_text(script_md(bs), encoding="utf-8")
    print("wrote film/story.json and docs/VIDEO_SCRIPT.md")


if __name__ == "__main__":
    main()
