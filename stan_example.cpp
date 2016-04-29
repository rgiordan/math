#include <stan/math.hpp>
#include <iostream>


// Compile with
// clang++ stan_example.cpp -I $STAN_HOME/src -isystem $STAN_HOME/lib/boost_1.55.0 -isystem $STAN_HOME/lib/eigen_3.2.4 -I $STAN_MATH_HOME -I /usr/local/include/eigen3/ -I /home/rgiordan/Documents/git_repos/math/lib/boost_1.58.0/

int main() {
  stan::math::var x, y, z;

  x = 2.5;
  y = 4.2;
  z = x * y + y * y * y;
  z -= 5.0;

  // Step 1
  std::vector<stan::math::var> independent_vars;
  independent_vars.push_back(x);
  independent_vars.push_back(y);

  // container for the gradients
  std::vector<double> gradients;

  // Step 2
  z.grad(independent_vars, gradients);

  // Step 3
  stan::math::recover_memory();

  std::cout << "z:       " << z.val() << std::endl
	    << "dz / dx: " << gradients[0] << std::endl
	    << "dz / dy: " << gradients[1] << std::endl;
  return 0;
}
