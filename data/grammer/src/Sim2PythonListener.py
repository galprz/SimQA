from typing import List

from grammer.src.SimCodeListener import SimCodeListener
from grammer.src.SimCodeParser import SimCodeParser


class Sim2PythonListener(SimCodeListener):
    def __init__(self, add_traces=False):
        self.reset_state()
        self.add_traces = add_traces

    def reset_state(self):
        self.vars = []
        self._indent_level = 0
        self._generated_code = ""

    def enterSimulation_fn(self, ctx: SimCodeParser.Simulation_fnContext):
        self._write_python_line("def simulation():")
        super().enterSimulation_fn(ctx)

    def enterOpen_code_block(self, ctx: SimCodeParser.Open_code_blockContext):
        self._indent_level += 1
        super().enterOpen_code_block(ctx)

    def enterClose_code_block(self, ctx: SimCodeParser.Close_code_blockContext):
        self._indent_level -= 1
        super().enterClose_code_block(ctx)

    def enterRepeat_stmt(self, ctx: SimCodeParser.Repeat_stmtContext):
        repeat_count = ctx.children[2].getText()
        self._write_python_line(f"for _ in range({repeat_count}):")
        super().enterRepeat_stmt(ctx)

    def enterIf_stmt(self, ctx: SimCodeParser.If_stmtContext):
        condition_txt = ctx.children[2].getText()
        self._write_python_line(f"if {condition_txt}:")
        super().enterIf_stmt(ctx)

    def enterAssignment_stmt(self, ctx: SimCodeParser.Assignment_stmtContext):
        new_candidate = ctx.children[0].getText()
        if self.add_traces and new_candidate not in self.vars:
            self.vars.append(new_candidate)
        assigment_txt = ctx.getText()[:-1]
        self._write_python_line(assigment_txt)
        super().enterAssignment_stmt(ctx)

    def enterReturn_stmt(self, ctx: SimCodeParser.Return_stmtContext):
        value = ctx.children[1].getText()
        if self.add_traces:
            self._write_python_line('print("return=' + value + '")')
        self._write_python_line(f"return {value}")
        super().enterReturn_stmt(ctx)

    def _get_var_state(self):
        vars_state = ",".join([var_name + "={" + var_name + "}" for var_name in self.vars])
        return 'print(f"' + vars_state + '")'

    def _write_python_line(self, line: str):
        for i in range(self._indent_level):
            self._generated_code += "\t"
        self._generated_code += line
        if self.add_traces and self.vars != [] and line[:3] != "for" and line[:2] != "if":
            self._generated_code += "\n"
            for i in range(self._indent_level):
                self._generated_code += "\t"
            self._generated_code += self._get_var_state()
        self._generated_code += "\n"

    def get_generated_code(self):
        return self._generated_code
