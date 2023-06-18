from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
print('rank {}.'.format(rank))
# M=2
# for i in range(-M, 0):
#     print(i)