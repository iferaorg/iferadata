SET DEFINE OFF;
CREATE TABLE OPTION_SNAPSHOT_1M
(
  SYMBOL            VARCHAR2(10 BYTE)           NOT NULL,
  DATETIME          TIMESTAMP(3)                NOT NULL,
  EXP_DATE          DATE                        NOT NULL,
  STRIKE            NUMBER(9,3)                 NOT NULL,
  RIGHT             VARCHAR2(1 BYTE)            NOT NULL,
  BID               NUMBER(12,6),
  ASK               NUMBER(12,6),
  MIDPOINT          NUMBER(12,6),
  UNDERLYING_PRICE  NUMBER(18,8),
  OPEN_INTEREST     NUMBER
)
COMPRESS FOR OLTP
TABLESPACE IFERA
PCTFREE    0
INITRANS   1
MAXTRANS   255
STORAGE    (
            INITIAL          64M
            NEXT             64M
            MAXSIZE          UNLIMITED
            MINEXTENTS       1
            MAXEXTENTS       UNLIMITED
            PCTINCREASE      0
            BUFFER_POOL      DEFAULT
           )
PARTITION BY LIST (SYMBOL) AUTOMATIC (
    PARTITION SPXW VALUES ('SPXW')
    TABLESPACE IFERA
    PCTFREE    0
    INITRANS   1
    MAXTRANS   255
    STORAGE    (
                INITIAL          64M
                NEXT             64M
                MAXSIZE          UNLIMITED
                MINEXTENTS       1
                MAXEXTENTS       UNLIMITED
                PCTINCREASE      0
                BUFFER_POOL      DEFAULT
            )
)
PARALLEL ( DEGREE DEFAULT INSTANCES DEFAULT )
ENABLE ROW MOVEMENT
/


CREATE UNIQUE INDEX OPTION_SNAPSHOT_1M_PK ON OPTION_SNAPSHOT_1M
(SYMBOL, DATETIME, EXP_DATE, STRIKE, RIGHT)
  PCTFREE    10
  INITRANS   2
  MAXTRANS   255
  STORAGE    (
              INITIAL          64M
              NEXT             64M
              MAXSIZE          UNLIMITED
              MAXEXTENTS       UNLIMITED
              PCTINCREASE      0
              BUFFER_POOL      DEFAULT
             )
LOCAL (  
  PARTITION
    TABLESPACE IFERA
    INITRANS   2
    MAXTRANS   255
    STORAGE    (
                INITIAL          64M
                NEXT             64M
                MAXSIZE          UNLIMITED
                MINEXTENTS       1
                MAXEXTENTS       UNLIMITED
                BUFFER_POOL      DEFAULT
               )
)
PARALLEL ( DEGREE DEFAULT INSTANCES DEFAULT )
COMPRESS 3
/

ALTER TABLE OPTION_SNAPSHOT_1M ADD (
  CONSTRAINT OPTION_SNAPSHOT_1M_PK
  PRIMARY KEY
  (SYMBOL, DATETIME, EXP_DATE, STRIKE, RIGHT)
  USING INDEX LOCAL
  ENABLE VALIDATE)
/
