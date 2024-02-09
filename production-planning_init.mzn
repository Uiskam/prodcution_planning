include "alldifferent.mzn";

% how many types of items do we produce
int: nbTypes;
set of int: Types = 1..nbTypes;
set of int: Types0 = 0..nbTypes;

% for how many timeslots do we plan the production
int: nbTimeslots;
set of int: TimeSlots = 1..nbTimeslots;

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
array[Items] of TimeSlots: dueTimeslots;

array[Items] of var TimeSlots: productionDate;

constraint forall ([productionDate[i] <= dueTimeslots[i] | i in Items]);
constraint alldifferent(productionDate);

constraint forall(i in Items)(abs(dueTimeslots[i] - productionDate[i]) <= 29);
solve satisfy;

output ["productionDate: \(productionDate)"];
