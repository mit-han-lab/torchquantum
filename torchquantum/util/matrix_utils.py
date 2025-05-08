import sys
import traceback
import numpy as np
import scipy.linalg




def ultra_precise_unitary(matrix, iterations=5, tolerance=1e-10):
    """
    Create an extremely precise unitary matrix from input matrix.
    Used to prevent 'TwoQubitWeylDecomposition: failed to diagonalize M2' errors.
    
    Args:
        matrix: Input matrix (should be approximately unitary)
        iterations: Number of refinement iterations
        tolerance: Target tolerance for unitarity (default: 1e-10)
    
    Returns:
        Ultra-precise unitary matrix with improved numerical properties
    """
    print(f"\n==== ULTRA_PRECISE_UNITARY DEBUG ====")
    print(f"Input matrix shape: {matrix.shape}")
    
    # Check initial unitarity
    input_deviation = np.max(np.abs(np.conjugate(matrix.T) @ matrix - np.eye(matrix.shape[0])))
    print(f"Input matrix deviation from unitarity: {input_deviation}")
    
    # If the input is already very unitary, just do a standard SVD cleanup
    if input_deviation < tolerance:
        print(f"Input already meets tolerance target of {tolerance}")
        return matrix
    
    # Store the best matrix and its deviation
    best_matrix = matrix.copy()
    best_deviation = input_deviation
    
    # Initial SVD decomposition - this generally gives good results
    V, s, Wh = scipy.linalg.svd(matrix, full_matrices=True, lapack_driver='gesvd')
    print(f"SVD singular values: {s}")
    # Force perfect singular values (exactly 1.0)
    s_unitary = np.ones_like(s)
    U = V @ np.diag(s_unitary) @ Wh
    
    # Check deviation after initial SVD
    deviation = np.max(np.abs(U.conj().T @ U - np.eye(U.shape[0])))
    print(f"After initial SVD, deviation: {deviation}")
    
    # If SVD immediately got us to tolerance level, return it
    if deviation < tolerance:
        print(f"Reached target tolerance with initial SVD")
        return U
    
    # Update best if SVD improved it
    if deviation < best_deviation:
        best_matrix = U.copy()
        best_deviation = deviation
    
    # Multiple refinement iterations trying different techniques
    for i in range(iterations):
        if best_deviation < tolerance:
            print(f"Reached target tolerance at iteration {i}")
            break
            
        print(f"Iteration {i+1}:")
        
        # Method 1: Polar decomposition
        try:
            H = U.conj().T @ U
            eigenvals, eigenvecs = scipy.linalg.eigh(H)
            print(f"  H eigenvalues: {eigenvals}")
            H_sqrt_inv = eigenvecs @ np.diag(1.0/np.sqrt(eigenvals)) @ eigenvecs.conj().T
            U_refined = U @ H_sqrt_inv
            
            new_deviation = np.max(np.abs(U_refined.conj().T @ U_refined - np.eye(U.shape[0])))
            print(f"  After polar decomposition, deviation: {new_deviation}")
            
            if new_deviation < best_deviation:
                best_matrix = U_refined.copy()
                best_deviation = new_deviation
                print(f"  Improved with polar decomposition")
                U = U_refined
        except Exception as e:
            print(f"  Polar decomposition failed: {str(e)}")
        
        # Method 2: Gram-Schmidt orthogonalization
        try:
            Q, R = scipy.linalg.qr(U, mode='economic')
            # Apply phase correction to maintain similarity to original matrix
            phases = np.diag(np.sign(np.diag(R)))
            U_gs = Q @ phases
            
            new_deviation = np.max(np.abs(U_gs.conj().T @ U_gs - np.eye(U.shape[0])))
            print(f"  After Gram-Schmidt, deviation: {new_deviation}")
            
            if new_deviation < best_deviation:
                best_matrix = U_gs.copy()
                best_deviation = new_deviation
                print(f"  Improved with Gram-Schmidt")
                U = U_gs
        except Exception as e:
            print(f"  Gram-Schmidt failed: {str(e)}")
            
        # Method 3: Direct normalization of columns
        try:
            U_norm = U.copy()
            for j in range(U.shape[1]):
                U_norm[:, j] = U[:, j] / np.sqrt(np.sum(np.abs(U[:, j])**2))
            
            new_deviation = np.max(np.abs(U_norm.conj().T @ U_norm - np.eye(U.shape[0])))
            print(f"  After column normalization, deviation: {new_deviation}")
            
            if new_deviation < best_deviation:
                best_matrix = U_norm.copy()
                best_deviation = new_deviation
                print(f"  Improved with column normalization")
                U = U_norm
        except Exception as e:
            print(f"  Column normalization failed: {str(e)}")
            
        # Method 4: Use double precision SVD
        if i == iterations-1 and best_deviation > tolerance:
            try:
                print("  Attempting high-precision SVD for final refinement")
                matrix_dp = np.array(best_matrix, dtype=np.complex128)
                V_dp, _, Wh_dp = scipy.linalg.svd(matrix_dp, full_matrices=True, lapack_driver='gesdd')
                U_dp = V_dp @ Wh_dp
                
                new_deviation = np.max(np.abs(U_dp.conj().T @ U_dp - np.eye(U_dp.shape[0])))
                print(f"  After high-precision SVD, deviation: {new_deviation}")
                
                if new_deviation < best_deviation:
                    best_matrix = U_dp.copy()
                    best_deviation = new_deviation
                    print(f"  Improved with high-precision SVD")
            except Exception as e:
                print(f"  High-precision SVD failed: {str(e)}")
    
    # Check if we've made the unitarity worse compared to input
    if best_deviation > input_deviation:
        print(f"WARNING: Processing made unitarity worse! Using original matrix.")
        best_matrix = matrix
        best_deviation = input_deviation
    
    # Final check if we've met the tolerance
    if best_deviation > tolerance:
        print(f"WARNING: Failed to achieve target tolerance of {tolerance}")
        # One final attempt with raw SVD which usually gives good results
        try:
            V, _, Wh = scipy.linalg.svd(matrix, full_matrices=True, lapack_driver='gesdd')
            U_final = V @ Wh
            final_deviation = np.max(np.abs(U_final.conj().T @ U_final - np.eye(U_final.shape[0])))
            if final_deviation < best_deviation:
                best_matrix = U_final
                best_deviation = final_deviation
                print(f"Final SVD improved deviation to {best_deviation}")
        except Exception:
            pass
    
    # Convert to high precision complex type
    final_matrix = np.array(best_matrix, dtype=np.complex128)
    final_deviation = np.max(np.abs(final_matrix.conj().T @ final_matrix - np.eye(final_matrix.shape[0])))
    print(f"Final deviation from unitarity: {final_deviation}")
    print(f"==== END ULTRA_PRECISE_UNITARY DEBUG ====\n")
    
    return final_matrix