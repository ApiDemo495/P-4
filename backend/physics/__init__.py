"""The thermodynamic capital layer (Round T).

A second, independent view of the BTC/PAXG pair built from physical law and
microstructural mathematics rather than from price patterns:

* Section 1  Landauer thermal valve        (``landauer``)
* Section 2  planetary solar flux          (``solar``)
* Section 4  work-to-rest-mass ratio       (``energy_mass``)
* Section 5  AMM price surface             (``amm``)
* Section 8  VPIN, fragmentation, O-U mean reversion, pendulum, A-S spread
                                           (``micro``)
* Section 10 PAXG / wBTC peg drift         (``peg``)
* Section 11/12 Kelly blend, thermodynamic clamp, TSR, phase angle
                                           (``unified``)

It is **one weighted layer in fusion** (``weight_physics``), not a hard
override: the 22 formulas and the Drosophila brain still decide, this layer
votes beside the AI agents.  Sections 3, 7 and 9 of the source document
(multi-node ZK Nash swarm, lending-protocol liquidation sniping, MEV block
sequencing) need infrastructure a single terminal cannot have and are not
implemented; nothing here claims a guaranteed yield.

Every number is tagged with its provenance: ``live`` (fetched from a public
API), ``tape`` (derived from the frozen market snapshot) or ``model`` (a
published-magnitude constant or a fallback estimate).
"""
