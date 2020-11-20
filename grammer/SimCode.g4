grammar SimCode;

parse
: ( simulation_fn | error ) EOF;

error
 : UNEXPECTED_CHAR
   {
raise RuntimeError("UNEXPECTED_CHAR=" + $UNEXPECTED_CHAR.text);
   }
 ;

simulation_fn
: FUNC SIMULATION_KEYWORD OPEN_PAR CLOSE_PAR open_code_block exp+ close_code_block;

exp: assignment_stmt | if_stmt | repeat_stmt | return_stmt;

return_stmt: RETURN (identifier | POSITIVE_INT | negative_int | ZERO | NUMERIC_LITERAL) SCOL ;

repeat_stmt: REPEAT OPEN_PAR (POSITIVE_INT | identifier) CLOSE_PAR open_code_block exp+ close_code_block;

if_stmt: IF OPEN_PAR condition_expression  CLOSE_PAR  open_code_block exp+ close_code_block ;

comperation_operator :  LT | GT | LT_EQ | GT_EQ | EQ | NOT_EQ1;

condition_expression: (identifier comperation_operator (POSITIVE_INT | NUMERIC_LITERAL | negative_int | ZERO)) |
 (identifier comperation_operator identifier);

assignment_stmt: identifier ASSIGN (numeric_expression) SCOL;

numeric_expression: (identifier operator identifier) |
            (identifier operator (POSITIVE_INT | NUMERIC_LITERAL)) |
            ((POSITIVE_INT | NUMERIC_LITERAL | negative_int ) operator identifier) |
            ((POSITIVE_INT | NUMERIC_LITERAL | negative_int ) operator (POSITIVE_INT | NUMERIC_LITERAL )) |
            (POSITIVE_INT | negative_int | NUMERIC_LITERAL | ZERO);

operator: PLUS | MINUS | STAR | DIV | INT_DIV;

open_code_block: OPEN_CURLY_PAR;

close_code_block: CLOSE_CURLY_PAR;

identifier: IDENTIFIER;

FUNC: F U N C;

SIMULATION_KEYWORD: S I M U L A T I O N;

IF: I F;

ELSE: E L S E;

REPEAT:  R E P E A T;

RETURN: R E T U R N;
ZERO: '0';
SCOL : ';';
DOT : '.';
OPEN_PAR : '(';
CLOSE_PAR : ')';
OPEN_CURLY_PAR: '{';
CLOSE_CURLY_PAR: '}';
COMMA : ',';
ASSIGN : '=';
STAR : '*';
PLUS : '+';
MINUS : '-';
INT_DIV : '//';
DIV : '/';
LT : '<';
LT_EQ : '<=';
GT : '>';
GT_EQ : '>=';
EQ : '==';
NOT_EQ1 : '!=';
COLON : ':';
NUMERIC_LITERAL: DIGIT+  '.' DIGIT+;

POSITIVE_INT: DIGIT+;
negative_int: MINUS DIGIT+;

IDENTIFIER : [a-zA-Z][a-zA-Z0-9_]* ;

SPACES
 : [ \u000B\t\r\n]+ -> channel(HIDDEN)
 ;

UNEXPECTED_CHAR
 : .
 ;

fragment DIGIT : [0-9];
fragment A : 'a';
fragment B : 'b';
fragment C : 'c';
fragment D : 'd';
fragment E : 'e';
fragment F : 'f';
fragment G : 'g';
fragment H : 'h';
fragment I : 'i';
fragment J : 'j';
fragment K : 'k';
fragment L : 'l';
fragment M : 'm';
fragment N : 'n';
fragment O : 'o';
fragment P : 'p';
fragment Q : 'q';
fragment R : 'r';
fragment S : 's';
fragment T : 't';
fragment U : 'u';
fragment V : 'v';
fragment W : 'w';
fragment X : 'x';
fragment Y : 'y';
fragment Z : 'z';