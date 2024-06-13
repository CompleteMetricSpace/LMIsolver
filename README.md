### A linear matrix inequality solver written in native Julia

Implementation of the two famous LMI algorithms:

1. The projection-based algorithm of Nemirovksi and Gahinet as presented originally in

   *Nemirovskii, Arkadii, and Pascal Gahinet. "The projective method for solving linear matrix inequalities." Proceedings of 1994 American Control Conference-ACC'94. Vol. 1. IEEE, 1994.*

2. An interior point method with logarithmic barrier as found in

   *Vandenberghe, Lieven, and Stephen Boyd. "Semidefinite programming." SIAM review 38.1 (1996): 49-95.*

Note that both solvers do not exploit the Block-matrix structure (or any other spare properties of the LMI's) and thus is only for demonstrational purposes.
More information can be obtained in

*El Ghaoui, Laurent, and Silviu-lulian Niculescu, eds. Advances in linear matrix inequality methods in control. Society for Industrial and Applied Mathematics, 2000.*

