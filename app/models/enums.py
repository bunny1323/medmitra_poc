from enum import Enum

class AgeGroup(str, Enum):
    ADULT = "adult"
    CHILD = "child"

class TopicGroup(str, Enum):
    ADULT_CORE = "adult_core"
    ADULT_EXTENDED = "adult_extended"
    PEDIATRIC = "pediatric"

class RelevanceLabel(str, Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"

class ChunkType(str, Enum):
    PARAGRAPH = "paragraph"
    HEADING = "heading"
    TABLE_LIKE = "table_like"
    LIST = "list"
    HEADING_CONTEXT = "heading_context"

class EmergencySeverity(str, Enum):
    URGENT = "urgent"
