include "alldifferent.mzn";
include "count.mzn";

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


array[Items] of 0..nbTimeslots: initProductionDate;


array[Items] of var Timeslots: productionDate;
array[Timeslots] of var Types0: solution;
%array[Types,Timeslots] of var 0..1: whenTypes;
var int: totalStorageCost;
var int: totalModeChangeCost;
var int: sMCC1;
var int: sMCC2;
var int: sMCC3;

var int: obj;

% *********** %
% Constraints %
% *********** %
%constraint forall(i in Timeslots)(sum(j in Items)(whenTypes[j,i]) <= 1);
%constraint forall(i in Items)(sum(j in Timeslots)(whenTypes[i,j]) == count(types,i));

constraint forall (i in Items) (
  initProductionDate[i] != 0 -> productionDate[i] == initProductionDate[i] 
);

constraint obj <= 27420;
constraint forall ([productionDate[i] <= dueTimeslots[i] | i in Items]);
constraint alldifferent(productionDate);
%constraint forall(i in Types)(count(solution,i,count(types,i)));
constraint count(solution,1,count(types,1));
constraint count(solution,2,count(types,2));
constraint count(solution,3,count(types,3));
constraint count(solution,4,count(types,4));
constraint count(solution,5,count(types,5));
constraint count(solution,6,count(types,6));
constraint count(solution,7,count(types,7));
constraint count(solution,8,count(types,8));
constraint count(solution,9,count(types,9));
constraint count(solution,10,count(types,10));
constraint count(solution,11,count(types,11));
constraint count(solution,12,count(types,12));
constraint count(solution,13,count(types,13));
constraint count(solution,14,count(types,14));
constraint count(solution,15,count(types,15));

constraint forall ([solution[productionDate[i]] == types[i] | i in Items]);

constraint totalStorageCost == sum([(dueTimeslots[i] - productionDate[i]) * storageCost | i in Items]);

constraint sMCC1 = sum([changeCost[solution[i-1], solution[i]] | i in 2..nbTimeslots]);
constraint sMCC2 = sum([changeCost[solution[i-2], solution[i]] | i in 3..nbTimeslots where solution[i-1] == 0]);
constraint sMCC3 = sum([changeCost[solution[i-3], solution[i]] | i in 4..nbTimeslots where (solution[i-1] == 0 /\ solution[i - 2] == 0)]);
constraint totalModeChangeCost = sMCC1 + sMCC2 + sMCC3;

constraint obj = totalStorageCost + totalModeChangeCost;
solve 
:: seq_search([
  int_search(productionDate, dom_w_deg, indomain_split ),
  int_search(solution, dom_w_deg, indomain_split ),
])
minimize obj;

output 
  [ "prod_dates: ", show([productionDate[i] | i in Items]), ";\n"] ++
  [ "obj = ", show(obj), ";\n"]
