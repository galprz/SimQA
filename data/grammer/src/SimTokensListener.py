from grammer.src.SimCodeListener import SimCodeListener
from grammer.src.SimCodeParser import SimCodeParser


class Sim2TokensListener(SimCodeListener):
    def __init__(self):
        super(Sim2TokensListener, self).__init__()
        self.reset_state()

    def reset_state(self):
        self._tokens = []

    def _seprate_to_tokens(self, children):
        return [token.getText() for token in children]

    def enterRepeat_stmt(self, ctx: SimCodeParser.Repeat_stmtContext):
        self._tokens += self._seprate_to_tokens(ctx.children[:5])
        super().enterRepeat_stmt(ctx)

    def exitRepeat_stmt(self, ctx: SimCodeParser.Repeat_stmtContext):
        self._tokens += ["}"]
        super().exitRepeat_stmt(ctx)

    def exitIf_stmt(self, ctx: SimCodeParser.If_stmtContext):
        self._tokens += ["}"]
        super().exitIf_stmt(ctx)

    def enterIf_stmt(self, ctx: SimCodeParser.If_stmtContext):
        self._tokens += self._seprate_to_tokens(ctx.children[:2])
        super().enterIf_stmt(ctx)

    def enterCondition_expression(self, ctx: SimCodeParser.Condition_expressionContext):
        self._tokens += self._seprate_to_tokens(ctx.children)
        super().enterCondition_expression(ctx)

    def exitCondition_expression(self, ctx: SimCodeParser.Condition_expressionContext):
        self._tokens += [")", "{"]
        super().exitCondition_expression(ctx)

    def enterReturn_stmt(self, ctx: SimCodeParser.Return_stmtContext):
        self._tokens += self._seprate_to_tokens(ctx.children)
        super().enterReturn_stmt(ctx)

    def enterAssignment_stmt(self, ctx: SimCodeParser.Assignment_stmtContext):
        self._tokens += self._seprate_to_tokens(ctx.children[:2])
        super().enterAssignment_stmt(ctx)

    def exitAssignment_stmt(self, ctx: SimCodeParser.Assignment_stmtContext):
        self._tokens += [";"]
        super().exitAssignment_stmt(ctx)

    def get_tokens(self):
        return self._tokens

    def enterNumeric_expression(self, ctx: SimCodeParser.Numeric_expressionContext):
        self._tokens += self._seprate_to_tokens(ctx.children[:5])
        super().enterNumeric_expression(ctx)
