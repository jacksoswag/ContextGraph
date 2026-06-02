# Hyperedge Containment Measurement (Layer 2)

8 nested facts, fresh store, top-12 mesh. Content recall = fraction of the inner fact's {subject, object} present in the mesh when seeding the outer subject.

| outer seed | relation | reified fact | recall OFF (w_hyper=0) | recall ON |
|---|---|---|---|---|
| scientist | believe | smoking cause cancer | 0.00 | 1.00 |
| government | announce | economy enter recession | 0.00 | 1.00 |
| study | show | exercise reduce mortality | 0.00 | 1.00 |
| court | rule | law violate constitution | 0.00 | 1.00 |
| report | confirm | company breach contract | 0.00 | 1.00 |
| teacher | explain | gravity bend spacetime | 0.00 | 1.00 |
| witness | claim | driver ignore signal | 0.00 | 1.00 |
| model | predict | warming raise sea level | 0.00 | 1.00 |

**Content recall:** 0.000 (containment OFF) → 1.000 (ON). Off-state recall is ~0 because the inner fact's nodes are unreachable from the outer subject by k-hop alone; the hyperedge binding is what makes the reified fact's content part of the gather.
