# Minimal PLM computational core

## Architectural pivot after CASM-X18R

CASM-X1 through X18R supports a smaller and clearer architecture than the original broad TAC-style system.

The immediate model target is not a full language model. It is a minimal persistent/executable computational core:

`observations -> construct computational roles -> instantiate working state -> execute reusable dynamics -> outcome`.

CASM already provides strong controlled evidence for explicit working state and shared local transition dynamics. The unresolved problem is construction: what computational state should exist, and how can the model extend that structure when new roles are required?

## Core state

Represent working state conceptually as

`W_t = {(r_i, tau_i, v_i, k_i)}_{i=1..N_t}`

where:

- `r_i` is computational role identity;
- `tau_i` is type;
- `v_i` is current executable value;
- `k_i` is structural/context information;
- `N_t` is the number of instantiated roles.

Role identity is not storage location. A role describes *what computational entity exists*; a physical slot or memory address describes only where its current state is stored.

## Surviving components

1. **Input/entity encoder** — initially supplied by controlled CASM data; later learned from observations/language.
2. **Recursive role generator / constructor** — next frontier. Generates extensible computational identities.
3. **Explicit working-state store** — strongly supported by X1-X5.
4. **Shared transition kernel** — strongly supported by X1-X3; no separate learned operator bank is currently justified.
5. **Resource controller** — X13-X18R show resource mechanisms can alter/stabilize organization, but they do not create extensible role structure. Their future job is to price reuse/creation/compute/storage.
6. **Controller/program induction** — later selects which operation/state transition should occur; current CASM still supplies commands.
7. **Verifier/repair** — later, once structural decisions themselves can be wrong; informed by REAL/TAC but not required in the current core.
8. **Persistent structure memory** — later, after learned structures can be created reliably within an episode.

## Target equations

Observation encoding:

`O = E_theta(x)`

Role construction:

`R_0 = G_theta(O, M)`

`R_{k+1} = Extend_theta(R_k, O, M)`

Working-state instantiation:

`W_0 = Instantiate(R, O)`

Control and execution:

`c_t = C_theta(W_t, O, M)`

`W_{t+1} = T_theta(W_t, c_t)`

Resource state:

`rho_{t+1} = U_theta(rho_t, W_t, c_t)`

Decode / verify / consolidate:

`y = D_theta(W_T)`

`v = V(y, W_T, O)`

`M_{k+1} = K(M_k, R, W_{0:T}, v)`

Only mechanisms that survive controlled falsifiers should enter the eventual PLM-v0.

## Generalization scorecard

Treat the following as separate axes:

1. **Execution depth** — can a learned transition run longer? Strong controlled evidence.
2. **Operation composition** — can known operations occur in unseen sequences? Strong controlled evidence.
3. **Ontology binding** — can known entities self-organize into latent executable state? Strong through X8 under fixed ontology.
4. **Structural extension** — can new computational roles be generated beyond the training ontology/horizon? Unsolved; X19 target.
5. **Program induction** — can the model choose which computation to execute? Future.
6. **Cross-episode persistence** — can useful learned structure survive, be retrieved, and improve later tasks? Future.

## Development sequence

- **X19:** recursive role generation; separate role identity from storage location.
- **X20:** dynamic state instantiation; remove fixed pre-existing storage identities.
- **X21:** learned controller/program induction; stop supplying the execution command sequence.
- **X22:** persistent reusable structure memory across episodes.
- **X23:** compositional retrieval/reuse of learned structures.
- **X24:** verifier-guided structural repair.
- **PLM-v0:** integrate only surviving mechanisms with a small pretrained language backbone used primarily for language↔structured-interface conversion.

Do not scale to a large language model before recursive role generation, dynamic working-state construction, and shared execution are independently validated.
