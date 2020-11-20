from typing import List

from grammer.src.SimCodeListener import SimCodeListener
from grammer.src.SimCodeParser import SimCodeParser

class SimPrinterListener(SimCodeListener):
    def __init__(self):
        self.reset_state()

    def reset_state(self):
        self._indent_level = 0
        self._generated_code = ""

    def enterSimulation_fn(self, ctx: SimCodeParser.Simulation_fnContext):
        self._write_python_line("func simulation(){")
        super().enterSimulation_fn(ctx)

    def enterOpen_code_block(self, ctx: SimCodeParser.Open_code_blockContext):
        self._indent_level += 1
        super().enterOpen_code_block(ctx)

    def enterClose_code_block(self, ctx: SimCodeParser.Close_code_blockContext):
        self._indent_level -= 1
        self._write_python_line("}")
        super().enterClose_code_block(ctx)

    def enterRepeat_stmt(self, ctx: SimCodeParser.Repeat_stmtContext):
        repeat_count = ctx.children[2].getText()
        self._write_python_line(f"repeat({repeat_count}){{")
        super().enterRepeat_stmt(ctx)

    def enterIf_stmt(self, ctx: SimCodeParser.If_stmtContext):
        condition_txt = ctx.children[2].getText()
        self._write_python_line(f"if({condition_txt}){{")
        super().enterIf_stmt(ctx)

    def enterAssignment_stmt(self, ctx: SimCodeParser.Assignment_stmtContext):
        self._write_python_line(ctx.getText())
        super().enterAssignment_stmt(ctx)

    def enterReturn_stmt(self, ctx: SimCodeParser.Return_stmtContext):
        value = ctx.children[1].getText()
        self._write_python_line(f"return {value};")
        super().enterReturn_stmt(ctx)

    def _write_python_line(self, line: str):
        for i in range(self._indent_level):
            self._generated_code += "\t"
        self._generated_code += line
        self._generated_code += "\n"

    def get_generated_code(self):
        return self._generated_code
