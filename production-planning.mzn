include "alldifferent.mzn";

% how many types of items do we produce
int: nbTypes;
set of int: Types = 1..nbTypes;
set of int: Types0 = 0..nbTypes;

% for how many timeslots do we plan the production
int: nbTimeslots;
set of int: Timeslots = 1..nbTimeslots;

% how many items do we have to produce
int: nbItems;
set of int: Items = 1..nbItems;

% how much does cost storage for a single timeslot
int: storageCost;

% how much costs reconfiguration of the assembly line
array[Types, Types] of int: changeCost;

% what are the types of each item
array[Items] of Types:  types;

% what are the deadlines of each item
array[Items] of Timeslots: dueTimeslots;

array[Items] of var Timeslots: productionDate;

% *********** %
% Constraints %
% *********** %

% deadline constraint
constraint forall ([productionDate[i] <= dueTimeslots[i] | i in Items]);
constraint alldifferent(productionDate);

% compute solution
array[Timeslots] of var Types0: solution;
constraint forall ([solution[productionDate[i]] == types[i] | i in Items]);

% computing cost fucntion
var int: totalStorageCost = sum([(dueTimeslots[i] - productionDate[i]) * storageCost | i in Items]);

% calculate total mode change cost
% take all pairs of days between which the production is stopped
% and sum the transition costs
% % knowing change[a, a] == 0
var 0..infinity: totalModeChangeCost = sum([
    changeCost[solution[i], solution[j]] 
    | i in Timeslots, j in Timeslots
    where (
      i < j
      /\ solution[i] != 0
      /\ solution[j] != 0
      /\ forall(k in (i+1)..(j-1)) (solution[k] == 0)
    )
]);


% objective function
var int: obj = totalStorageCost + totalModeChangeCost;


% indomain_random OP af.
% dom_w_deg gives similar times to first_fail, rest are garbage.
solve 
:: seq_search([
  int_search(solution, first_fail, indomain_min),
  int_search(productionDate, dom_w_deg, indomain_random)
])
minimize obj;

output 
  [ "solution = ", show([solution[i] - 1 | i in Timeslots]), ";\n"] ++
  [ "obj = ", show(obj), ";\n"]
