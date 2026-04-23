# Keeps track of phases, so they run sequentially with no
PHASE_IDLE     = 0  # waiting for user input
PHASE_RESEARCH = 1  # Scrapers active, Mind ingesting connections
PHASE_EXPORT   = 2  # main.py → Mind: write agents_pre_group.json
PHASE_EXPORTED = 3  # Mind → main.py: file ready, run grouping script
PHASE_IMPORT   = 5  # main.py → Mind: read agent_mapping.json
PHASE_IMPORTED = 6  # Mind → main.py: merge complete
PHASE_PHYSICS  = 7  # Physics engine running
PHASE_STABLE   = 8  # Physics exited, graph finalized

# Seeded logical connectors currently recognized by d_logic_extractor.py.
REL_IDENTITY    = 0
REL_CONDITIONAL = 1
REL_SUBSET      = 2
LOGICAL_CONNECTORS = ["be", "lead to", "subset of"]
