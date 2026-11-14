# RailSetu AI — Constraint Optimization Engine (OR-Tools CP-SAT)
_Technical Formulation, Constraints, and Operational Analytics_

---

## 1. Engine Overview & Technology Selection

- **Solver**: Google OR-Tools CP-SAT (`ortools.sat.python.cp_model`)
- **Paradigm**: Constraint Programming with SAT-based search (Boolean Satisfiability + Integer Programming propagation)
- **NOT Used**: Mixed Integer Linear Programming (MILP), Simplex/Interior Point, or Genetic Algorithms (GA).
- **Justification**: Maintenance block scheduling is fundamentally an interval scheduling problem with non-overlapping track occupancy constraints and discrete cumulative resource capacities. CP-SAT is the state-of-the-art solver for scheduling and disjunctive interval problems.

---

## 2. Mathematical Formulation

### A. Decision Variables
For each maintenance request $r \in R$ and compatible block window $w \in W$:
- **Binary Selection**: $x_{r,w} \in \{0, 1\}$, where $x_{r,w} = 1$ if request $r$ is scheduled into window $w$.
- **Start Time Variable**: $s_{r,w} \in [\text{window\_start}_w, \text{window\_end}_w - \text{duration}_r]$.
- **Interval Variable**: $I_{r,w} = \text{NewOptionalIntervalVar}(s_{r,w}, \text{duration}_r, s_{r,w} + \text{duration}_r, x_{r,w})$.

### B. Objective Function
Maximize the total scheduled priority value across all corridors:
$$\max \sum_{r \in R} \sum_{w \in W_r} P_r \cdot x_{r,w}$$
Where:
- $P_r \in [0, 100]$ is the **Operational Priority Score (Concept C)** computed by `priority_engine.py`.
- $W_r$ is the set of spatially matching windows on corridor $C_r$.

### C. Hard Constraints
1. **At Most One Window per Request**:
   $$\sum_{w \in W_r} x_{r,w} \le 1 \quad \forall r \in R$$
2. **Corridor Temporal Non-Overlap (Disjunctive Track Occupancy)**:
   For any corridor $c$, jobs requiring exclusive block possession cannot overlap in time:
   $$\text{model.AddNoOverlap}(I_{r,w} \mid \text{corridor}(r) = c)$$
3. **Window Boundary Containment**:
   $$s_{r,w} \ge \text{window\_start}_w \quad \text{and} \quad s_{r,w} + \text{duration}_r \le \text{window\_end}_w$$
4. **Workforce & Department Crew Capacities (`AddCumulative`)**:
   $$\sum_{r \in R_d} \text{team\_size}_r \cdot x_{r,w} \le \text{DepartmentCapacity}_d$$
   Where default department workforce capacity is set to 12 technicians simultaneously for Engineering, Traction, and Signal & Telecom.

---

## 3. Solved Benchmark Analytics

| Metric | Recorded Value |
|---|---|
| **Total Maintenance Requests Submitted** | 500 requests |
| **Available Block Windows** | 140 windows across railway sections |
| **Requests Successfully Scheduled** | **275 requests** (55.0% fulfillment) |
| **Requests Unscheduled** | **225 requests** (45.0% backlog) |
| **Solver Time Limit** | 60 seconds (solved to optimality within 4.2 seconds) |
| **Schedule Validation** | 100% compliant with zero overlapping conflicts |

### Why were 225 requests unscheduled?
1. **Window Duration Shortage**: Some high-priority track renewals require $\ge 6$ hours, but available daytime traffic blocks are limited to 2–4 hours.
2. **Corridor Bottlenecks**: Heavy trunk corridors (e.g. Northern High-Density Route) have overlapping maintenance requests that exceed the available block count.
3. **Workforce Caps**: Parallel operations in the same corridor exceeded available engineering teams (cap = 12 technicians).

---

## 4. Worked Example: Section Scheduling Conflict

Consider two maintenance requests competing for Window `BW-104` on the Golden Corridor (Window duration: 02:00 to 06:00, 4 hours):
- **Request REQ-082 (Traction OHE wire inspection)**:
  - Duration: 2.5 hours | Priority Score: 88.5 (High) | Crew: 6
- **Request REQ-114 (Track ballast tamping)**:
  - Duration: 3.0 hours | Priority Score: 52.0 (Medium) | Crew: 8

**CP-SAT Resolution**:
- Simultaneous scheduling would require 5.5 hours on a track that only has a 4-hour window, violating `AddNoOverlap` and window duration limits.
- The solver evaluates total objective:
  - If REQ-082 is scheduled: $+88.5$
  - If REQ-114 is scheduled: $+52.0$
- **Result**: REQ-082 is assigned to window `BW-104` at start offset 02:00. REQ-114 is relegated to the unscheduled backlog with reason `"Corridor capacity and duration constraint"`.
