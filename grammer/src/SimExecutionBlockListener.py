from enum import Enum

from grammer.src.SimCodeListener import SimCodeListener
from grammer.src.SimCodeParser import SimCodeParser


class BlockType(Enum):
    NotInBlock=1,
    IfBlock=2,
    WhileBlock=3,
    Assignment=4
    Return=5



class SimExecutionBlockListener(SimCodeListener):

    def enterReturn_stmt(self, ctx: SimCodeParser.Return_stmtContext):
        super().enterReturn_stmt(ctx)
        if self._current_block == BlockType.NotInBlock:
            self._current_block = BlockType.Return
            self._blocks.append(f"{ctx.children[0].getText()} {ctx.children[1].getText()};")

    def exitReturn_stmt(self, ctx: SimCodeParser.Return_stmtContext):
        super().exitReturn_stmt(ctx)
        if self._current_block == BlockType.Return:
            self._current_block = BlockType.NotInBlock

    def enterRepeat_stmt(self, ctx: SimCodeParser.Repeat_stmtContext):
        super().enterRepeat_stmt(ctx)
        if self._current_block == BlockType.NotInBlock:
            self._current_block = BlockType.WhileBlock
            self._blocks.append(ctx.getText())
        super().enterRepeat_stmt(ctx)

    def exitRepeat_stmt(self, ctx: SimCodeParser.Repeat_stmtContext):
        super().exitRepeat_stmt(ctx)
        if self._current_block == BlockType.WhileBlock:
            self._current_block = BlockType.NotInBlock

    def exitAssignment_stmt(self, ctx: SimCodeParser.Assignment_stmtContext):
        super().exitAssignment_stmt(ctx)
        if self._current_block == BlockType.Assignment:
            self._current_block = BlockType.NotInBlock

    def enterIf_stmt(self, ctx: SimCodeParser.If_stmtContext):
        if self._current_block == BlockType.NotInBlock:
            self._current_block=BlockType.IfBlock
            self._blocks.append(ctx.getText())
        super().enterIf_stmt(ctx)

    def exitIf_stmt(self, ctx: SimCodeParser.If_stmtContext):
        super().exitIf_stmt(ctx)
        if self._current_block == BlockType.IfBlock:
            self._current_block = BlockType.NotInBlock

    def enterAssignment_stmt(self, ctx:SimCodeParser.Assignment_stmtContext):
        if self._current_block == BlockType.NotInBlock:
            self._current_block = BlockType.Assignment
            self._blocks.append(ctx.getText())
        super().enterAssignment_stmt(ctx)

    def __init__(self):
        self._current_code = ""
        self._blocks = []
        self._current_block = BlockType.NotInBlock

    def get_blocks(self):
        return self._blocks
