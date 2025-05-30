import numpy as np
import scipy.optimize as opt


class TrilocalModel:
    """
    Possible signatures:
    TrilocalModel(p_alpha, p_beta, p_gamma, p_a, p_b, p_c)
    TrilocalModel(c_alpha, c_beta, c_gamma, ma, mb,mc, x)

    Constructs a trilocal model either:
    i) from the hidden variable distributions ``p_alpha``, ``p_beta``,
    ``p_gamma`` and response functions ``p_a``, ``p_b``, ``p_c``, or;
    ii) from the hidden variable cardinalities ``c_alpha``, ``c_beta``,
    ``c_gamma``, output cardinalities ``m_a``, ``m_b``, ``m_c`` and array
    of free parameters ``x``.
    """

    def __init__(self, *args):
        """
        Possible signatures:
        TrilocalModel(p_alpha, p_beta, p_gamma, p_a, p_b, p_c)
        TrilocalModel(c_alpha, c_beta, c_gamma, ma, mb,mc, x)

        Constructs a trilocal model either:
        i) from the hidden variable distributions ``p_alpha``, ``p_beta``,
        ``p_gamma`` and response functions ``p_a``, ``p_b``, ``p_c``, or;
        ii) from the hidden variable cardinalities ``c_alpha``, ``c_beta``,
        ``c_gamma``, output cardinalities ``m_a``, ``m_b``, ``m_c`` and array
        of free parameters ``x``.
        """
        if len(args) == 6:
            p_alpha, p_beta, p_gamma, p_a, p_b, p_c = args
            self.p_alpha = np.array(p_alpha).flatten()
            self.p_beta = np.array(p_beta).flatten()
            self.p_gamma = np.array(p_gamma).flatten()
            self.p_a = np.array(p_a)
            self.p_b = np.array(p_b)
            self.p_c = np.array(p_c)
        elif len(args) == 7:
            c_alpha, c_beta, c_gamma, ma, mb, mc, x = args
            x = np.array(x).flatten()
            end_alpha = c_alpha - 1
            end_beta = end_alpha + c_beta - 1
            end_gamma = end_beta + c_gamma - 1
            end_a = end_gamma + (ma - 1) * c_beta * c_gamma
            end_b = end_a + (mb - 1) * c_gamma * c_alpha
            # end_c = end_b + (mc - 1) * c_alpha * c_beta
            self.p_alpha, self.p_beta, self.p_gamma, self.p_a, self.p_b, \
                self.p_c = np.split(x, [end_alpha, end_beta, end_gamma,
                                        end_a, end_b])
            self.p_alpha = np.concatenate((self.p_alpha,
                                           [1 - np.sum(self.p_alpha)]))
            self.p_beta = np.concatenate((self.p_beta,
                                          [1 - np.sum(self.p_beta)]))
            self.p_gamma = np.concatenate((self.p_gamma,
                                           [1 - np.sum(self.p_gamma)]))
            self.p_a = self.p_a.reshape((ma - 1, c_beta, c_gamma))
            self.p_a = np.concatenate((self.p_a,
                                       1 - self.p_a.sum(axis=0, keepdims=True)),
                                      axis=0)
            self.p_b = self.p_b.reshape((mb - 1, c_gamma, c_alpha))
            self.p_b = np.concatenate((self.p_b,
                                       1 - self.p_b.sum(axis=0, keepdims=True)),
                                      axis=0)
            self.p_c = self.p_c.reshape((mc - 1, c_alpha, c_beta))
            self.p_c = np.concatenate((self.p_c,
                                       1 - self.p_c.sum(axis=0, keepdims=True)),
                                      axis=0)
        else:
            raise ValueError(f'Either 6 or 7 arguments expected. '
                             f'Got {len(args)} argument(s) instead.')
        self.update_cardinalities()

    def __str__(self):
        """
        Returns string representation of hidden variable distributions and
        response functions.
        """
        return f'p_alpha = {str(self.p_alpha)}\n' \
               f'p_beta  = {str(self.p_beta)}\n' \
               f'p_gamma = {str(self.p_gamma)}\n' \
               f'p_a =\n{str(self.p_a[0:-1, :, :])}\n' \
               f'p_b =\n{str(self.p_b[0:-1, :, :])}\n' \
               f'p_c =\n{str(self.p_c[0:-1, :, :])}'

    def show_hidden_variable_distributions(self):
        """
        Prints hidden variable distributions.
        """
        print(f'p_alpha = {str(self.p_alpha)}')
        print(f'p_beta  = {str(self.p_beta)}')
        print(f'p_gamma = {str(self.p_gamma)}')

    def show_response_functions(self, decimal_places=8):
        """
        Prints response functions.
        """
        print(f'p_a =\n{str(self.p_a[0:-1, :, :].round(decimal_places))}')
        print(f'p_b =\n{str(self.p_b[0:-1, :, :].round(decimal_places))}')
        print(f'p_c =\n{str(self.p_c[0:-1, :, :].round(decimal_places))}')

    def cardinalities(self):
        """
        Return tuple with hidden variable cardinalities ``c_alpha``,
        ``c_beta``, ``c_gamma`` and output cardinalities ``ma``, ``mb``, ``mc``.
        """
        return (len(self.p_alpha), len(self.p_beta), len(self.p_gamma),
                self.p_a.shape[0], self.p_b.shape[0], self.p_c.shape[0])

    def update_cardinalities(self):
        self.c_alpha, self.c_beta, self.c_gamma, self.ma, self.mb, self.mc = \
            self.cardinalities()

    def degrees_of_freedom(self):
        """
        Calculates the number of free parameters in the trilocal model.
        """
        return self.c_alpha + self.c_beta + self.c_gamma - 3 \
            + self.c_alpha * self.c_beta * (self.mc - 1) \
            + self.c_beta * self.c_gamma * (self.ma - 1) \
            + self.c_alpha * self.c_gamma * (self.mb - 1)

    def behavior(self):
        """
        Calculates the statistical behavior p(a,b,c) for the trilocal model.
        """
        # Array indices for np.einsum:
        #   p_alpha: alpha -> i
        #   p_beta: beta -> j
        #   p_gamma: gamma -> k
        #   p_a: a, beta, gamma -> ljk
        #   p_b: b, gamma, alpha -> mki
        #   p_c: c, alpha, beta -> nij
        #   px: a, b, c -> lmn
        return np.einsum('i,j,k,ljk,mki,nij->lmn',
                         self.p_alpha, self.p_beta, self.p_gamma,
                         self.p_a, self.p_b, self.p_c)

    def cost(self, p):
        """
        Calculates the sum of squared errors between the model behavior and a
        given target behavior ``p``.
        """
        return np.sum((self.behavior() - p) ** 2)

    @staticmethod
    def cost_for_optimizer(x, p, c_alpha, c_beta, c_gamma, ma, mb, mc):
        """
        Cost function for the optimizer.
        """
        return TrilocalModel(c_alpha, c_beta, c_gamma, ma, mb, mc, x).cost(p)

    @staticmethod
    def uniform(c_alpha, c_beta, c_gamma, ma, mb, mc):
        """
        Creates trilocal model with uniform probability distributions with
        cardinalities ``c_alpha``, ``c_beta``, ``c_gamma``, ``ma``, ``mb``,
        ``mc``.
        """
        p_alpha = 1 / c_alpha * np.ones(c_alpha)
        p_beta = 1 / c_beta * np.ones(c_beta)
        p_gamma = 1 / c_gamma * np.ones(c_gamma)
        p_a = 1 / ma * np.ones((ma, c_beta, c_gamma))
        p_b = 1 / mb * np.ones((mb, c_gamma, c_alpha))
        p_c = 1 / mc * np.ones((mc, c_alpha, c_beta))
        return TrilocalModel(p_alpha, p_beta, p_gamma, p_a, p_b, p_c)

    @staticmethod
    def random(c_alpha, c_beta, c_gamma, ma, mb, mc):
        """
        Creates random trilocal model with cardinalities ``c_alpha``,
        ``c_beta``, ``c_gamma``, ``ma``, ``mb``, ``mc``.
        """
        p_alpha = np.random.dirichlet(np.ones(c_alpha))
        p_beta = np.random.dirichlet(np.ones(c_beta))
        p_gamma = np.random.dirichlet(np.ones(c_gamma))
        p_a = np.moveaxis(np.random.dirichlet(np.ones(ma), (c_beta, c_gamma)),
                          -1, 0)
        p_b = np.moveaxis(np.random.dirichlet(np.ones(mb), (c_gamma, c_alpha)),
                          -1, 0)
        p_c = np.moveaxis(np.random.dirichlet(np.ones(mc), (c_alpha, c_beta)),
                          -1, 0)
        return TrilocalModel(p_alpha, p_beta, p_gamma, p_a, p_b, p_c)

    def optimizer_representation(self):
        """
        Returns representation consisting only of free parameters ``x``.
        Useful for the optimizer.
        """
        end_alpha = self.c_alpha - 1
        end_beta = end_alpha + self.c_beta - 1
        end_gamma = end_beta + self.c_gamma - 1
        end_a = end_gamma + (self.ma - 1) * self.c_beta * self.c_gamma
        end_b = end_a + (self.mb - 1) * self.c_gamma * self.c_alpha
        end_c = end_b + (self.mc - 1) * self.c_alpha * self.c_beta
        x = np.zeros(end_c)
        x[0:end_alpha] = self.p_alpha[0:end_alpha]
        x[end_alpha:end_beta] = self.p_beta[0:end_beta - end_alpha]
        x[end_beta:end_gamma] = self.p_gamma[0:end_gamma - end_beta]
        x[end_gamma:end_a] = self.p_a[0:self.ma - 1, :, :].flatten()
        x[end_a:end_b] = self.p_b[0:self.mb - 1, :, :].flatten()
        x[end_b:end_c] = self.p_c[0:self.mc - 1, :, :].flatten()
        return x

    def optimize(self, p, initial_guess=None, number_of_trials=1, tol=1e-4):
        """
        Returns model optimized to replicate given behavior ``p``.
        """
        dof = self.degrees_of_freedom()
        bounds = opt.Bounds(np.zeros(dof), np.ones(dof))
        # The code below assembles the hidden variable positivity constraints
        n_constraints = (3 + self.c_beta * self.c_gamma
                         + self.c_gamma * self.c_alpha
                         + self.c_alpha * self.c_beta)
        end_alpha = self.c_alpha - 1
        end_beta = end_alpha + self.c_beta - 1
        end_gamma = end_beta + self.c_gamma - 1
        coeffs = np.zeros(shape=(n_constraints, dof))
        coeffs[0, 0:end_alpha] = 1
        coeffs[1, end_alpha:end_beta] = 1
        coeffs[2, end_beta:end_gamma] = 1
        # The code below assembles the response function positivity constraints
        row_a = 3 + self.c_beta * self.c_gamma
        row_b = row_a + self.c_gamma * self.c_alpha
        row_c = row_b + self.c_alpha * self.c_beta
        end_a = end_gamma + (self.ma - 1) * self.c_beta * self.c_gamma
        end_b = end_a + (self.mb - 1) * self.c_gamma * self.c_alpha
        end_c = end_b + (self.mc - 1) * self.c_alpha * self.c_beta
        coeffs[3:row_a, end_gamma:end_a] = np.tile(np.eye(self.c_beta
                                                          * self.c_gamma),
                                                   self.ma - 1)
        coeffs[row_a:row_b, end_a:end_b] = np.tile(np.eye(self.c_gamma
                                                          * self.c_alpha),
                                                   self.mb - 1)
        coeffs[row_b:row_c, end_b:end_c] = np.tile(np.eye(self.c_alpha
                                                          * self.c_beta),
                                                   self.mc - 1)
        linear_constraints = opt.LinearConstraint(coeffs,
                                                  -np.inf * np.ones(row_c),
                                                  np.ones(row_c))
        if initial_guess is None:
            x0 = TrilocalModel.random(self.c_alpha, self.c_beta, self.c_gamma,
                                      self.ma, self.mb, self.mc) \
                .optimizer_representation()
        else:
            x0 = initial_guess
        # ---------------------------------------------------------------------
        optimized_model = self
        error = np.sqrt(self.cost(p) / (self.ma * self.mb * self.mc))
        for i in range(number_of_trials):
            print(f'Trial {i + 1}. ', end='')
            solution = opt.minimize(TrilocalModel.cost_for_optimizer,
                                    x0,
                                    args=(p, self.c_alpha, self.c_beta,
                                          self.c_gamma, self.ma, self.mb,
                                          self.mc),
                                    method='trust-constr',
                                    constraints=linear_constraints,
                                    options={'verbose': 1}, bounds=bounds)
            partial_model = TrilocalModel(self.c_alpha, self.c_beta,
                                          self.c_gamma, self.ma, self.mb,
                                          self.mc, solution.x)
            partial_error = np.sqrt(partial_model.cost(p) / (self.ma * self.mb
                                                             * self.mc))
            if partial_error < error:
                optimized_model = partial_model
                error = partial_error
            if partial_error < tol:
                break
            # x0 = TrilocalModel.random(self.c_alpha, self.c_beta, self.c_gamma,
            #                           self.ma, self.mb, self.mc) \
            #     .optimizer_representation()
            x0 = np.random.random(dof)
            x0[0:end_alpha] = 1 / self.c_alpha
            x0[end_alpha:end_beta] = 1 / self.c_beta
            x0[end_beta:end_gamma] = 1 / self.c_gamma
        return optimized_model

    def optimize_e3(self, e1, e2, initial_guess=None, number_of_trials=1, tol=1e-4):
        """
        Returns model that minimizes e3 subject to fixed e1 and e2.
        """
        dof = self.degrees_of_freedom()
        bounds = opt.Bounds(np.zeros(dof), np.ones(dof))
        # The code below assembles the hidden variable positivity constraints
        n_constraints = (3 + self.c_beta * self.c_gamma
                         + self.c_gamma * self.c_alpha
                         + self.c_alpha * self.c_beta)
        end_alpha = self.c_alpha - 1
        end_beta = end_alpha + self.c_beta - 1
        end_gamma = end_beta + self.c_gamma - 1
        coeffs = np.zeros(shape=(n_constraints, dof))
        coeffs[0, 0:end_alpha] = 1
        coeffs[1, end_alpha:end_beta] = 1
        coeffs[2, end_beta:end_gamma] = 1
        # The code below assembles the response function positivity constraints
        row_a = 3 + self.c_beta * self.c_gamma
        row_b = row_a + self.c_gamma * self.c_alpha
        row_c = row_b + self.c_alpha * self.c_beta
        end_a = end_gamma + (self.ma - 1) * self.c_beta * self.c_gamma
        end_b = end_a + (self.mb - 1) * self.c_gamma * self.c_alpha
        end_c = end_b + (self.mc - 1) * self.c_alpha * self.c_beta
        coeffs[3:row_a, end_gamma:end_a] = np.tile(np.eye(self.c_beta
                                                          * self.c_gamma),
                                                   self.ma - 1)
        coeffs[row_a:row_b, end_a:end_b] = np.tile(np.eye(self.c_gamma
                                                          * self.c_alpha),
                                                   self.mb - 1)
        coeffs[row_b:row_c, end_b:end_c] = np.tile(np.eye(self.c_alpha
                                                          * self.c_beta),
                                                   self.mc - 1)
        linear_constraints = opt.LinearConstraint(coeffs,
                                                  -np.inf * np.ones(row_c),
                                                  np.ones(row_c))

        # The code below assembles the equality constraints on e1 and e2
        def fun(x):
            model = TrilocalModel(self.c_alpha, self.c_beta, self.c_gamma, self.ma, self.mb, self.mc, x)
            _, ea, eb, ec = model.e1()
            _, eab, ebc, eac = model.e2()
            return np.array([ea, eb, ec, eab, ebc, eac])
        rhs = np.array([e1, e1, e1, e2, e2, e2])
        equality_constraints = opt.NonlinearConstraint(fun, rhs, rhs)

        # Cost function
        def cost(x):
            model = TrilocalModel(self.c_alpha, self.c_beta, self.c_gamma, self.ma, self.mb, self.mc, x)
            return model.e3()

        # Initial guess
        if initial_guess is None:
            x0 = TrilocalModel.random(self.c_alpha, self.c_beta, self.c_gamma,
                                      self.ma, self.mb, self.mc) \
                .optimizer_representation()
        else:
            x0 = initial_guess
        # ---------------------------------------------------------------------
        optimized_model = self
        e3 = np.inf
        for i in range(number_of_trials):
            print(f'Trial {i + 1}. ', end='')
            solution = opt.minimize(cost, x0,
                                    method='trust-constr',
                                    constraints=[linear_constraints,
                                                 equality_constraints],
                                    options={'verbose': 1, 'maxiter': 40000},
                                    bounds=bounds)
            partial_model = TrilocalModel(self.c_alpha, self.c_beta,
                                          self.c_gamma, self.ma, self.mb,
                                          self.mc, solution.x)
            partial_e3 = partial_model.e3()
            if partial_e3 < e3:
                optimized_model = partial_model
                e3 = partial_e3
            x0 = np.random.random(dof)
            x0[0:end_alpha] = 1 / self.c_alpha
            x0[end_alpha:end_beta] = 1 / self.c_beta
            x0[end_beta:end_gamma] = 1 / self.c_gamma
        return optimized_model

    def relabel_hidden_variable(self, variable, new_labels):
        """
        Relabels the hidden variable indicated by the integer parameter
        ``variable``, 0 being alpha, 1 being beta and 2 being gamma. The new
        labels are indicated by the array ``new_labels``, which must be an array
        containing the integers 0, 1, ..., c-1, where c is the cardinality of
        the hidden variable being relabelled.
        """
        if variable == 0:
            self.p_alpha = self.p_alpha.take(new_labels)
            self.p_b = self.p_b.take(new_labels, axis=2)
            self.p_c = self.p_c.take(new_labels, axis=1)
        elif variable == 1:
            self.p_beta = self.p_beta.take(new_labels)
            self.p_a = self.p_a.take(new_labels, axis=1)
            self.p_c = self.p_c.take(new_labels, axis=2)
        elif variable == 2:
            self.p_gamma = self.p_gamma.take(new_labels)
            self.p_a = self.p_a.take(new_labels, axis=2)
            self.p_b = self.p_b.take(new_labels, axis=1)

    def remove_hidden_variable_labels(self, variable, labels):
        """
        Removes labels for the hidden variable indicated by the integer
        parameter ``variable``, 0 being alpha, 1 being beta and 2 being gamma.
        The labels to be removed are indicated by the array ``labels``, which
        must be an array of integers in the set {0, 1, ..., c-1}, where c is
        the cardinality of the hidden variable being altered.
        """
        if variable == 0:
            self.p_alpha = np.delete(self.p_alpha, labels)
            self.p_alpha = self.p_alpha / self.p_alpha.sum()
            self.p_b = np.delete(self.p_b, labels, axis=2)
            self.p_c = np.delete(self.p_c, labels, axis=1)
        elif variable == 1:
            self.p_beta = np.delete(self.p_beta, labels)
            self.p_beta = self.p_beta / self.p_beta.sum()
            self.p_a = np.delete(self.p_a, labels, axis=1)
            self.p_c = np.delete(self.p_c, labels, axis=2)
        elif variable == 2:
            self.p_gamma = np.delete(self.p_gamma, labels)
            self.p_gamma = self.p_gamma / self.p_gamma.sum()
            self.p_a = np.delete(self.p_a, labels, axis=2)
            self.p_b = np.delete(self.p_b, labels, axis=1)
        self.update_cardinalities()

    def relabel_output(self, party, new_labels):
        """
        Relabels the output indicated by the integer parameter ``variable``,
        0 being a, 1 being b and 2 being c. The new labels are indicated by the
        array ``new_labels``, which must be an array containing the integers
        0, 1, ..., m-1, where m is the cardinality of the output being
        relabelled.
        """
        if party == 0:
            self.p_a = self.p_a.take(new_labels, axis=0)
        elif party == 1:
            self.p_b = self.p_b.take(new_labels, axis=0)
        elif party == 2:
            self.p_c = self.p_c.take(new_labels, axis=0)

    def exchange_hidden_variables(self, variable1, variable2):
        """
        Exchange the hidden variables indicated by the integer parameters
        ``variable1`` and ``variable2``, 0 being alpha, 1 being beta and
        2 being gamma.
        """
        if {variable1, variable2} == {0, 1}:
            self.p_alpha, self.p_beta = self.p_beta, self.p_alpha
            self.p_a, self.p_b = (self.p_b.swapaxes(1, 2),
                                  self.p_a.swapaxes(1, 2))
            self.p_c = self.p_c.swapaxes(1, 2)
        elif {variable1, variable2} == {1, 2}:
            self.p_beta, self.p_gamma = self.p_gamma, self.p_beta
            self.p_b, self.p_c = (self.p_c.swapaxes(1, 2),
                                  self.p_b.swapaxes(1, 2))
            self.p_a = self.p_a.swapaxes(1, 2)
        elif {variable1, variable2} == {0, 2}:
            self.p_alpha, self.p_gamma = self.p_gamma, self.p_alpha
            self.p_a, self.p_c = (self.p_c.swapaxes(1, 2),
                                  self.p_a.swapaxes(1, 2))
            self.p_b = self.p_b.swapaxes(1, 2)
        self.update_cardinalities()

    def exchange_parties(self, party1, party2):
        """
        Exchange the parties indicated by the integer parameters ``party1``
        and ``party2``, 0 being a, 1 being b and 2 being c.
        """
        self.exchange_hidden_variables(party1, party2)

    def standardize(self, exchange_parties_allowed=True):
        """
        Relabels hidden variables so that probabilities are listed in
        descending order. Also, if the exchange of parties is allowed, reorder
        hidden variables so that c_alpha >= c_beta >= c_gamma.
        """
        # Rearrange hidden variables in descending cardinalities
        if exchange_parties_allowed:
            sorted_variables = self.sort_hidden_variables()
            if sorted_variables == (0, 1, 2):  # a b c
                self.exchange_hidden_variables(0, 2)
            elif sorted_variables == (0, 2, 1):  # a c b
                self.exchange_hidden_variables(0, 1)
                self.exchange_hidden_variables(1, 2)
            elif sorted_variables == (1, 0, 2):  # b a c
                self.exchange_hidden_variables(0, 2)
                self.exchange_hidden_variables(1, 2)
            elif sorted_variables == (1, 2, 0):  # b c a
                self.exchange_hidden_variables(1, 2)
            elif sorted_variables == (2, 0, 1):  # c a b
                self.exchange_hidden_variables(0, 1)
        # Rearrange hidden variable labels in descending probabilities
        self.relabel_hidden_variable(0, np.argsort(-self.p_alpha))
        self.relabel_hidden_variable(1, np.argsort(-self.p_beta))
        self.relabel_hidden_variable(2, np.argsort(-self.p_gamma))

    def round(self, tol=1e-4):
        """
        Removes hidden variable labels with probabilities less than ``tol``.
        """
        self.remove_hidden_variable_labels(0, self.p_alpha < tol)
        self.remove_hidden_variable_labels(1, self.p_beta < tol)
        self.remove_hidden_variable_labels(2, self.p_gamma < tol)

    def sort_hidden_variables(self):
        """
        Return indices that would sort the hidden variables with respect to
        their cardinalities (0 represents alpha, 1 represents beta, 2
        represents gamma). In case the variables have the same cardinalities,
        the sorting is done with respect to information entropy.
        """
        # I have looked for a function f(c,s) (c is the cardinality and s is
        # the entropy) satisfying the following properties:
        # c1 > c2 -> f(c1,s1) > f(c2,s2)
        # s1 > s2 -> f(c,s1) > f(c,s2)
        # Any function satisfying these could be used to sort the hidden
        # variables in the way we need. One such function is
        # f(c,s) = c + tanh(s), so that is what I use in the code below.
        s_alpha, s_beta, s_gamma = self.hidden_variable_entropies()
        return tuple(np.argsort((self.c_alpha + np.tanh(s_alpha),
                                 self.c_beta + np.tanh(s_beta),
                                 self.c_gamma + np.tanh(s_gamma))))

    def hidden_variable_entropies(self):
        return ((-p * np.log2(p)).sum() for p in (self.p_alpha, self.p_beta,
                                                  self.p_gamma))

    def e1(self, atol=0.01):
        """
        Computes the average correlator `e1` and individual correlators `ea`, 
        `eb`, and `ec` from the model behavior. The method also checks if the 
        deviations between the average and individual correlators are within a 
        specified tolerance. If it is not, the distribution is not symmetric, 
        and it doesn't really make sense to speak of a single one-body 
        correlator, so `e1` is assigned the `None` object.
    
        The behavior distribution is expected to be a 2x2x2 array representing the 
        probabilities of measurement outcomes for three parties (A, B, C). The correlators 
        are calculated as follows:
        - `ea`: Difference between the sum of probabilities for outcomes where A=0 and A=1.
        - `eb`: Difference between the sum of probabilities for outcomes where B=0 and B=1.
        - `ec`: Difference between the sum of probabilities for outcomes where C=0 and C=1.
        - `e1`: Average of `ea`, `eb`, and `ec`.
    
        Parameters:
        -----------
        atol : float, optional
            Absolute tolerance for the maximum deviation between the average correlator `e1` 
            and the individual correlators `ea`, `eb`, and `ec` (default is 0.01).
    
        Returns:
        --------
        tuple
            - If the maximum deviation is within the tolerance, returns a tuple containing:
                - `e1` (float): The average correlator.
                - `ea` (float): The correlator for party A.
                - `eb` (float): The correlator for party B.
                - `ec` (float): The correlator for party C.
            - If the maximum deviation exceeds the tolerance, returns a tuple containing:
                - `None`: Indicates that the deviation is too large.
                - `ea` (float): The correlator for party A.
                - `eb` (float): The correlator for party B.
                - `ec` (float): The correlator for party C.
    
        Notes:
        ------
        - The behavior distribution is obtained by calling the `behavior()` method of the class.
        - If the behavior distribution does not have the shape (2, 2, 2), the method returns `None` 
          for all correlators.
    
        Examples:
        ---------
        >>> # Assuming `self.behavior()` returns a valid 2x2x2 probability distribution
        >>> result = self.e1(atol=0.01)
        >>> if result[0] is not None:
        ...     print(f"Average correlator e1: {result[0]}")
        ... else:
        ...     print("Deviation too large. Individual correlators:", result[1:])
        """
        p = self.behavior()
        if p.shape != (2, 2, 2):
            return None
        ea = p[0, :, :].sum() - p[1, :, :].sum()
        eb = p[:, 0, :].sum() - p[:, 1, :].sum()
        ec = p[:, :, 0].sum() - p[:, :, 1].sum()
        e1 = (ea + eb + ec) / 3
        max_dev = max(np.abs(e1 - ea), np.abs(e1 - eb), np.abs(e1 - ec))
        if max_dev < atol:
            return e1, ea, eb, ec
        else:
            return None, ea, eb, ec

    def e2(self, atol=0.01):
        """
        Computes the average correlator `e2` and individual two-party 
        correlators `eab`, `ebc`, and `eac` from the model behavior. The method 
        also checks if the deviations between the average and individual 
        correlators are within a specified tolerance. If it is not, the distribution 
        is not symmetric and it doesn't really make sense to speak of a single 
        two-body correlator, so `e2` is assigned the `None` object.
    
        The behavior distribution is expected to be a 2x2x2 array representing the 
        probabilities of measurement outcomes for three parties (A, B, C). The correlators 
        are calculated as follows:
        - `eab`: Correlator for parties A and B.
        - `ebc`: Correlator for parties B and C.
        - `eac`: Correlator for parties A and C.
        - `e2`: Average of `eab`, `ebc`, and `eac`.
    
        Parameters:
        -----------
        atol : float, optional
            Absolute tolerance for the maximum deviation between the average correlator `e2` 
            and the individual correlators `eab`, `ebc`, and `eac` (default is 0.01).
    
        Returns:
        --------
        tuple
            - If the maximum deviation is within the tolerance, returns a tuple containing:
                - `e2` (float): The average correlator.
                - `eab` (float): The correlator for parties A and B.
                - `ebc` (float): The correlator for parties B and C.
                - `eac` (float): The correlator for parties A and C.
            - If the maximum deviation exceeds the tolerance, returns a tuple containing:
                - `None`: Indicates that the deviation is too large.
                - `eab` (float): The correlator for parties A and B.
                - `ebc` (float): The correlator for parties B and C.
                - `eac` (float): The correlator for parties A and C.
    
        Notes:
        ------
        - The behavior distribution is obtained by calling the `behavior()` method of the class.
        - If the behavior distribution does not have the shape (2, 2, 2), the method returns `None` 
          for all correlators.
    
        Examples:
        ---------
        >>> # Assuming `self.behavior()` returns a valid 2x2x2 probability distribution
        >>> result = self.e2(atol=0.01)
        >>> if result[0] is not None:
        ...     print(f"Average correlator e2: {result[0]}")
        ... else:
        ...     print("Deviation too large. Individual correlators:", result[1:])
        """
        p = self.behavior()
        if p.shape != (2, 2, 2):
            return None
        eab = p[0, 0, :].sum() - p[0, 1, :].sum() - p[1, 0, :].sum() \
            + p[1, 1, :].sum()
        ebc = p[:, 0, 0].sum() - p[:, 0, 1].sum() - p[:, 1, 0].sum() \
            + p[:, 1, 1].sum()
        eac = p[0, :, 0].sum() - p[0, :, 1].sum() - p[1, :, 0].sum() \
            + p[1, :, 1].sum()
        e2 = (eab + ebc + eac) / 3
        max_dev = max(np.abs(e2 - eab), np.abs(e2 - ebc), np.abs(e2 - eac))
        if max_dev < atol:
            return e2, eab, ebc, eac
        else:
            return None, eab, ebc, eac

    def e3(self):
        p = self.behavior()
        if p.shape != (2, 2, 2):
            return None
        abc = np.array([[[1, -1],
                         [-1, 1]],
                        [[-1, 1],
                         [1, -1]]])
        return (p*abc).sum()
