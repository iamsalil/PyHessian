#*
# @file Different utility functions
# Copyright (c) Zhewei Yao, Amir Gholami
# All rights reserved.
# This file is part of PyHessian library.
#
# PyHessian is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# PyHessian is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with PyHessian.  If not, see <http://www.gnu.org/licenses/>.
#*

import torch
import math
from torch.autograd import Variable
import numpy as np
import os
import time, datetime

from pyhessian.utils import group_product, norm, multi_add, group_add, normalization, get_params_grad, hessian_vector_product, orthnormal


class hessian():
    """
    The class used to compute :
        i) the top 1 (n) eigenvalue(s) of the neural network
        ii) the trace of the entire neural network
        iii) the estimated eigenvalue density
    """

    def __init__(self, model, criterion, data=None, dataloader=None, cuda=True, data_save_dir="data", record_data=False):
        """
        model: the model that needs Hessain information
        criterion: the loss function
        data: a single batch of data, including inputs and its corresponding labels
        dataloader: the data loader including bunch of batches of data
        """

        # make sure we either pass a single batch or a dataloader
        assert (data != None and dataloader == None) or (data == None and
                                                         dataloader != None)

        self.model = model.eval()  # make model is in evaluation model
        self.criterion = criterion

        if data != None:
            self.data = data
            self.full_dataset = False
        else:
            self.data = dataloader
            self.full_dataset = True

        if cuda:
            self.device = 'cuda'
        else:
            self.device = 'cpu'

        # checking whether or not data save folder has been created
        self.record_data = record_data
        self.data_save_dir = data_save_dir + "/"
        if record_data and (not os.path.exists(data_save_dir)):
            try:
                os.mkdir(data_save_dir)
            except OSError:
                print("Could not create data save directory")

        # pre-processing for single batch case to simplify the computation.
        if not self.full_dataset:
            self.inputs, self.targets = self.data
            if self.device == 'cuda':
                self.inputs, self.targets = self.inputs.cuda(
                ), self.targets.cuda()

            # if we only compute the Hessian information for a single batch data, we can re-use the gradients.
            outputs = self.model(self.inputs)
            loss = self.criterion(outputs, self.targets)
            loss.backward(create_graph=True)

        # this step is used to extract the parameters from the model
        params, gradsH = get_params_grad(self.model)
        self.params = params
        self.gradsH = gradsH  # gradient used for Hessian computation

    def dataloader_hv_product(self, v):

        device = self.device
        num_data = 0  # count the number of datum points in the dataloader

        THv = [torch.randn(p.size()).to(device) for p in self.params
              ]  # accumulate result
        for inputs, targets in self.data:
            self.model.zero_grad()
            tmp_num_data = inputs.size(0)
            outputs = self.model(inputs.to(device))
            loss = self.criterion(outputs, targets.to(device))
            loss.backward(create_graph=True)
            params, gradsH = get_params_grad(self.model)
            self.model.zero_grad()
            Hv = torch.autograd.grad(gradsH,
                                     params,
                                     grad_outputs=v,
                                     only_inputs=True,
                                     retain_graph=False)
            THv = [
                THv1 + Hv1 * float(tmp_num_data) + 0.
                for THv1, Hv1 in zip(THv, Hv)
            ]
            num_data += float(tmp_num_data)

        THv = [THv1 / float(num_data) for THv1 in THv]
        eigenvalue = group_product(THv, v).cpu().item()
        return eigenvalue, THv

    def eigenvalues(self, maxIter=100, tol=1e-3, top_n=1, debug=False):
        """
        compute the top_n eigenvalues using power iteration method
        maxIter: maximum iterations used to compute each single eigenvalue
        tol: the relative tolerance between two consecutive eigenvalue computations from power iteration
        top_n: top top_n eigenvalues will be computed
        """

        assert top_n >= 1

        device = self.device

        eigenvalues = []
        eigenvectors = []

        computed_dim = 0

        # Prepare to record data
        if self.record_data:
            now = datetime.datetime.now()
            timestamp = "_{:02d}{:02d}_{:02d}{:02d}{:02d}".format(now.day, now.month, now.hour, now.minute, now.second)
            save_file = self.data_save_dir + "TopEigen" + timestamp + ".txt"
            total_time_to_compute = []
            iters_to_compute = []

        start_time = time.time()
        while computed_dim < top_n:
            if debug:
                print("Computing eigenvalue #{}".format(computed_dim+1))
            eigenvalue = None
            v = [torch.randn(p.size()).to(device) for p in self.params
                ]  # generate random vector
            v = normalization(v)  # normalize the vector

            for i in range(maxIter):
                if debug:
                    print("   Iteration {}".format(i))
                v = orthnormal(v, eigenvectors)
                self.model.zero_grad()

                if self.full_dataset:
                    tmp_eigenvalue, Hv = self.dataloader_hv_product(v)
                else:
                    Hv = hessian_vector_product(self.gradsH, self.params, v)
                    tmp_eigenvalue = group_product(Hv, v).cpu().item()

                v = normalization(Hv)

                if eigenvalue == None:
                    eigenvalue = tmp_eigenvalue
                else:
                    if abs(eigenvalue - tmp_eigenvalue) / (abs(eigenvalue) +
                                                           1e-6) < tol:
                        break
                    else:
                        eigenvalue = tmp_eigenvalue
            # Record data
            total_time_to_compute.append(time.time() - start_time);
            iters_to_compute.append(i)
            eigenvalues.append(eigenvalue)
            eigenvectors.append(v)
            computed_dim += 1
        # Write data if applicable
        if self.record_data:
            with open(save_file, 'w') as f:
                f.write("Eigenvalue\tTotal Elapsed Time(s)\t#Iterations\n")
                for i in range(top_n):
                    f.write("{}\t{}\t{}\n".format(i+1, total_time_to_compute[i], iters_to_compute[i]))
        return eigenvalues, eigenvectors

    def trace(self, maxIter=100, tol=1e-3, debug=False):
        """
        compute the trace of hessian using Hutchinson's method
        maxIter: maximum iterations used to compute trace
        tol: the relative tolerance
        """

        device = self.device
        trace_vhv = []
        trace = 0.

        # Prepare to record data
        if self.record_data:
            now = datetime.datetime.now()
            timestamp = "_{:02d}{:02d}_{:02d}{:02d}{:02d}".format(now.day, now.month, now.hour, now.minute, now.second)
            save_file = self.data_save_dir + "Trace" + timestamp + ".txt"
            total_time_to_compute = []
            trace_estimate = []

        start_time = time.time()
        for i in range(maxIter):
            if debug:
                    print("Iteration {}".format(i))
            self.model.zero_grad()
            v = [
                torch.randint_like(p, high=2, device=device)
                for p in self.params
            ]
            # generate Rademacher random variables
            for v_i in v:
                v_i[v_i == 0] = -1

            if self.full_dataset:
                _, Hv = self.dataloader_hv_product(v)
            else:
                Hv = hessian_vector_product(self.gradsH, self.params, v)
            trace_vhv.append(group_product(Hv, v).cpu().item())

            total_time_to_compute.append(time.time() - start_time)
            trace_estimate.append(np.mean(trace_vhv))

            if abs(np.mean(trace_vhv) - trace) / (trace + 1e-6) < tol:
                # Write data if applicable
                if self.record_data:
                    with open(save_file, 'w') as f:
                        f.write("Iteration\tTotal Elapsed Time(s)\tTrace Estimate\n")
                        for i in range(len(total_time_to_compute)):
                            f.write("{}\t{}\t{}\n".format(i+1, total_time_to_compute[i], trace_estimate[i]))
                return trace_vhv
            else:
                trace = np.mean(trace_vhv)
        # Trace could not converge
        # Write data if applicable
        if self.record_data:
            with open(save_file, 'w') as f:
                f.write("Iteration\tTotal Elapsed Time(s)\tTrace Estimate\n")
                for i in range(len(total_time_to_compute)):
                    f.write("{}\t{}\t{}\n".format(i+1, total_time_to_compute[i], trace_estimate[i]))
        return trace_vhv

    def density(self, iter=100, n_v=1, debug=False):
        """
        compute estimated eigenvalue density using stochastic lanczos algorithm (SLQ)
        iter: number of iterations used to compute trace
        n_v: number of SLQ runs

        """

        device = self.device
        eigen_list_full = []
        weight_list_full = []

        # Prepare to record data
        if self.record_data:
            now = datetime.datetime.now()
            timestamp = "_{:02d}{:02d}_{:02d}{:02d}{:02d}".format(now.day, now.month, now.hour, now.minute, now.second)
            save_file = self.data_save_dir + "ESD" + timestamp + ".txt"

        start_time = time.time();
        for k in range(n_v):
            v = [
                torch.randint_like(p, high=2, device=device)
                for p in self.params
            ]
            # generate Rademacher random variables
            for v_i in v:
                v_i[v_i == 0] = -1
            v = normalization(v)

            # standard lanczos algorithm initlization
            v_list = [v]
            w_list = []
            alpha_list = []
            beta_list = []
            ############### Lanczos
            for i in range(iter):
                if debug:
                    print("Iteration {}".format(i))
                self.model.zero_grad()
                w_prime = [torch.zeros(p.size()).to(device) for p in self.params]
                if i == 0:
                    if self.full_dataset:
                        _, w_prime = self.dataloader_hv_product(v)
                    else:
                        w_prime = hessian_vector_product(
                            self.gradsH, self.params, v)
                    alpha = group_product(w_prime, v)
                    alpha_list.append(alpha.cpu().item())
                    w = group_add(w_prime, v, alpha=-alpha)
                    w_list.append(w)
                else:
                    beta = torch.sqrt(group_product(w, w))
                    beta_list.append(beta.cpu().item())
                    if beta_list[-1] != 0.:
                        # We should re-orth it
                        v = orthnormal(w, v_list)
                        v_list.append(v)
                    else:
                        # generate a new vector
                        w = [torch.randn(p.size()).to(device) for p in self.params]
                        v = orthnormal(w, v_list)
                        v_list.append(v)
                    if self.full_dataset:
                        _, w_prime = self.dataloader_hv_product(v)
                    else:
                        w_prime = hessian_vector_product(
                            self.gradsH, self.params, v)
                    alpha = group_product(w_prime, v)
                    alpha_list.append(alpha.cpu().item())
                    w_tmp = group_add(w_prime, v, alpha=-alpha)
                    w = group_add(w_tmp, v_list[-2], alpha=-beta)

            T = torch.zeros(iter, iter).to(device)
            for i in range(len(alpha_list)):
                T[i, i] = alpha_list[i]
                if i < len(alpha_list) - 1:
                    T[i + 1, i] = beta_list[i]
                    T[i, i + 1] = beta_list[i]
            a_, b_ = torch.eig(T, eigenvectors=True)

            eigen_list = a_[:, 0]
            weight_list = b_[0, :]**2
            eigen_list_full.append(list(eigen_list.cpu().numpy()))
            weight_list_full.append(list(weight_list.cpu().numpy()))
        # Write data if applicable
        stop_time = time.time()
        if self.record_data:
            with open(save_file, 'w') as f:
                f.write("Total Elapsed Time(s)\n")
                f.write("{}\n".format(stop_time - start_time))
        return eigen_list_full, weight_list_full

    # CUSTOM FUNCTIONS BELOW HERE
    # todo: implement sketch, eigenvalues_lanczos, trace_forced_lengthy, scdf, density_to_scdf

    def sketch(self, d, debug=False):
        """
        Sketch function and scale down from a nxn matrix to a dxd matrix
        Do this by right multiplying by d nx1 column vectors to get a nxd matrix
            then left multiplying by a dxn matrix
        Sketch is made up of Rademacher variables (helps with trace calculation as well)
        Output is a dxd numpy array
        """

        device = self.device
        # Generate d Rademacher vectors v and calculate corresponding Hv
        print("starting")
        print("d = " + str(d))
        print(time.time())
        vs = []
        Hvs = []
        for i in range(d):
            print(i)
            # Generate Rademacher random variables
            v = [torch.randint_like(p, high=2, device=device) for p in self.params]
            for v_i in v:
                v_i[v_i == 0] = -1
            # print(norm(v))
            v = normalization(v)
            # print(norm(v))
            # print(group_product(v, v).cpu().item())
            vs.append(v)
            # Calculate Hv
            self.model.zero_grad()
            if self.full_dataset:
                _, Hv = self.dataloader_hv_product(v)
            else:
                Hv = hessian_vector_product(self.gradsH, self.params, v)
            Hvs.append(Hv)
        # Create sketched matrix template
        print(time.time())
        sketched_hessian = np.zeros((d, d))
        # Fill in matrix as A_ij = v_i' * Hv_j
        for i in range(d):
            for j in range(d):
                # print("({}, {})".format(i, j))
                sketched_hessian[i, j] = group_product(vs[i], Hvs[j]).cpu().item()/d
        print(time.time())
        return sketched_hessian

    def eigenvalues_lanczos(self, k, debug=False):
        """
        Compute the top k eigenvalues by Lacnzos Method for approximating eigenvalues
        """

        device = self.device

        # Prepare to record data
        if self.record_data:
            now = datetime.datetime.now()
            timestamp = "_{:02d}{:02d}_{:02d}{:02d}{:02d}".format(now.day, now.month, now.hour, now.minute, now.second)
            save_file = self.data_save_dir + "Lanczos" + timestamp + ".txt"
            total_time_to_compute = []

        start_time = time.time()
        # Pick a random first vector, making sure it has norm 1
        print("starting with q1")
        q0 = [torch.randn(p.size()).to(device) for p in self.params]
        q0 = normalization(q0)
        total_time_to_compute.append(time.time() - start_time)
        # Calculate Hq1
        self.model.zero_grad()
        if self.full_dataset:
            _, Hq0 = self.dataloader_hv_product(q0)
        else:
            Hq0 = hessian_vector_product(self.gradsH, self.params, q0)
        # First column
        qs = [q0]
        Hqs = [Hq0]
        T = np.zeros((k+1, k))
        T[0, 0] = group_product(qs[0], Hqs[0]).cpu().item()
        r = multi_add([Hqs[0], qs[0]], [1, -1*T[0, 0]]) # r = Hq0 - T00*q0
        T[1, 0] = norm(r) # T10 = |r|
        T[0, 1] = T[1, 0] # T symmetric
        q1 = [ri / T[1, 0] for ri in r] #q2 = r/|r|
        total_time_to_compute.append(time.time() - start_time)
        # Calculate Hq2
        self.model.zero_grad()
        if self.full_dataset:
            _, Hq1 = self.dataloader_hv_product(q1)
        else:
            Hq1 = hessian_vector_product(self.gradsH, self.params, q1)
        qs.append(q1)
        Hqs.append(Hq1)
        # Subsequent columns (columns 1 - k-1)
        for i in range(1, k):
            print(i)
            T[i, i] = group_product(qs[i], Hqs[i]).cpu().item()
            r = multi_add([Hqs[i], qs[i-1], qs[i]], [1, -1*T[i-1, i], -1*T[i, i]])
            T[i+1, i] = norm(r)
            if i != k-1:
                T[i, i+1] = T[i+1, i]
            q = [ri / T[i+1, i] for ri in r]
            total_time_to_compute.append(time.time() - start_time)
            self.model.zero_grad()
            if self.full_dataset:
                _, Hq = self.dataloader_hv_product(q)
            else:
                Hq = hessian_vector_product(self.gradsH, self.params, q)
            qs.append(q)
            Hqs.append(Hq)
        # print(T)
        T_UH = T[0:k, 0:k] #T_UH is square Upper Hessenberg
        # np.save("T_100", T)
        # print(T_UH)
        # print(np.linalg.eigvalsh(T_UH))

        # Write data if applicable
        if self.record_data:
            with open(save_file, 'w') as f:
                f.write("Total Elapsed Time(s)\tEigenvalues\n")
                for i in range(k):
                    eigs_i = np.linalg.eigvalsh(T[0:i, 0:i])
                    s = ""
                    for e in eigs_i:
                        s += "\t" + str(e)
                    s = str(total_time_to_compute[i]) + s + "\n"
                    f.write(s)
        return np.linalg.eigvalsh(T_UH)

    def trace_forced_lengthy(self, maxIter=150, num_reps=1, debug=False):
        """
        As a test, do not terminate trace calculation after 'convergence'
        Go a fixed number of iterations
        """

        device = self.device
        trace_vhv = []
        trace = 0.

        # Prepare to record data
        if self.record_data:
            now = datetime.datetime.now()
            timestamp = "_{:02d}{:02d}_{:02d}{:02d}{:02d}".format(now.day, now.month, now.hour, now.minute, now.second)
            save_file = self.data_save_dir + "Trace" + timestamp + ".txt"
            total_time_to_compute = []
            trace_estimate = []

        start_time = time.time()
        for i in range(maxIter):
            if debug:
                    print("Iteration {}".format(i))
            self.model.zero_grad()
            v = [
                torch.randint_like(p, high=2, device=device)
                for p in self.params
            ]
            # generate Rademacher random variables
            for v_i in v:
                v_i[v_i == 0] = -1

            if self.full_dataset:
                _, Hv = self.dataloader_hv_product(v)
            else:
                Hv = hessian_vector_product(self.gradsH, self.params, v)
            trace_vhv.append(group_product(Hv, v).cpu().item())

            total_time_to_compute.append(time.time() - start_time)
            trace_estimate.append(np.mean(trace_vhv))

            if abs(np.mean(trace_vhv) - trace) / (trace + 1e-6) < tol:
                # Write data if applicable
                if self.record_data:
                    with open(save_file, 'w') as f:
                        f.write("Iteration\tTotal Elapsed Time(s)\tTrace Estimate\n")
                        for i in range(len(total_time_to_compute)):
                            f.write("{}\t{}\t{}\n".format(i+1, total_time_to_compute[i], trace_estimate[i]))
                return trace_vhv
            else:
                trace = np.mean(trace_vhv)
        # Trace could not converge
        # Write data if applicable
        if self.record_data:
            with open(save_file, 'w') as f:
                f.write("Iteration\tTotal Elapsed Time(s)\tTrace Estimate\n")
                for i in range(len(total_time_to_compute)):
                    f.write("{}\t{}\t{}\n".format(i+1, total_time_to_compute[i], trace_estimate[i]))
        return trace_vhv

    def test_function(self):
        """
        test function
        """
        print("testing")
        device = self.device

        self.model.zero_grad()

        for p in self.params:
            print(p.size())

        # Generate Rademacher random variable
        v = [torch.randint_like(p, high=2, device=device) for p in self.params]
        for v_i in v:
            v_i[v_i == 0] = -1
        # Multiply with Hessian
        if self.full_dataset:
            _, Hv = self.dataloader_hv_product(v)
        else:
            Hv = hessian_vector_product(self.gradsH, self.params, v)
        #print(type(v))
        print(type(Hv))
        print(len(Hv))
        for hi in Hv:
            print(type(hi))
        #for vi in v:


        '''
        start_time = time.time()
        while computed_dim < top_n:
            if debug:
                print("Computing eigenvalue #{}".format(computed_dim+1))
            eigenvalue = None
            v = [torch.randn(p.size()).to(device) for p in self.params
                ]  # generate random vector
            v = normalization(v)  # normalize the vector

            for i in range(maxIter):
                if debug:
                    print("   Iteration {}".format(i))
                v = orthnormal(v, eigenvectors)
                self.model.zero_grad()

                if self.full_dataset:
                    tmp_eigenvalue, Hv = self.dataloader_hv_product(v)
                else:
                    Hv = hessian_vector_product(self.gradsH, self.params, v)
                    tmp_eigenvalue = group_product(Hv, v).cpu().item()

                v = normalization(Hv)

                if eigenvalue == None:
                    eigenvalue = tmp_eigenvalue
                else:
                    if abs(eigenvalue - tmp_eigenvalue) / (abs(eigenvalue) +
                                                           1e-6) < tol:
                        break
                    else:
                        eigenvalue = tmp_eigenvalue
            # Record data
            total_time_to_compute.append(time.time() - start_time);
            iters_to_compute.append(i)
            eigenvalues.append(eigenvalue)
            eigenvectors.append(v)
            computed_dim += 1
        # Write data if applicable
        if self.record_data:
            with open(save_file, 'w') as f:
                f.write("Eigenvalue\tTotal Elapsed Time(s)\t#Iterations\n")
                for i in range(top_n):
                    f.write("{}\t{}\t{}\n".format(i+1, total_time_to_compute[i], iters_to_compute[i]))
        return eigenvalues, eigenvectors
        '''
