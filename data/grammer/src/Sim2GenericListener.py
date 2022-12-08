from typing import List
import random
from grammer.src.SimCodeListener import SimCodeListener
from grammer.src.SimCodeParser import SimCodeParser


class Sim2GenericListener(SimCodeListener):
    def __init__(self):
        super(Sim2GenericListener, self).__init__()
        self.reset_state()

    def reset_state(self):
        self.available_vars = [f"VAR{i + 1}" for i in range(100)]
        self.available_vars.reverse()
        # should we shuffle the vars?
        # random.shuffle(self.available_vars)
        self.used_vars = {}
        self._generated_code = ""

    def _convert_var(self, var):
        if var in self.used_vars:
            converted_var = self.used_vars[var]
        else:
            converted_var = self.available_vars.pop()
            self.used_vars[var] = converted_var
        return converted_var

    def _convert_identifiers(self, children):
        if children is None:
            return []
        return [
            self._convert_var(item.getText()) if type(item) is SimCodeParser.IdentifierContext else item.getText()
            for item in children
        ]

    def enterRepeat_stmt(self, ctx: SimCodeParser.Repeat_stmtContext):
        txt = "".join(self._convert_identifiers(ctx.children[:5]))
        self._generated_code += txt
        super().enterRepeat_stmt(ctx)

    def exitRepeat_stmt(self, ctx: SimCodeParser.Repeat_stmtContext):
        self._generated_code += "}"
        super().exitRepeat_stmt(ctx)

    def enterIf_stmt(self, ctx: SimCodeParser.If_stmtContext):
        txt = "if(" + "".join(self._convert_identifiers(ctx.children[2].children)) + "){"
        self._generated_code += txt
        super().enterIf_stmt(ctx)

    def exitIf_stmt(self, ctx: SimCodeParser.If_stmtContext):
        self._generated_code += "}"
        super().exitIf_stmt(ctx)

    def enterReturn_stmt(self, ctx: SimCodeParser.Return_stmtContext):
        txt = " ".join(self._convert_identifiers(ctx.children))
        self._generated_code += txt
        super().enterReturn_stmt(ctx)

    def enterAssignment_stmt(self, ctx: SimCodeParser.Assignment_stmtContext):
        if len(ctx.children) < 3:
            return
        txt = (
            "".join(self._convert_identifiers(ctx.children[:2]))
            + "".join(self._convert_identifiers(ctx.children[2].children))
            + ctx.children[3].getText()
        )
        self._generated_code += txt
        super().enterAssignment_stmt(ctx)

    def get_generated_code(self):
        return self._generated_code
