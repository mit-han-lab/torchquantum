from torchquantum.density import density_mat
from torchquantum.density import density_func




if __name__ == "__main__":
    mat = density_func.mat_dict["hadamard"]
    print(mat)
    D = density_mat.DensityMatrix(2)
    D.print_2d(0)
    newD=density_func.apply_unitary_density_bmm(D._matrix,mat,[0])

    D.update_matrix(newD)

    D.print_2d(0)
