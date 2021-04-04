import argparse
from datetime import datetime, timedelta

from qiskit import IBMQ, execute
from qiskit.tools.monitor import job_monitor
from qiskit import QuantumCircuit
# from qiskit.providers.ibmq import least_busy
from qiskit.providers.ibmq.ibmqbackend import IBMQBackend
from torchpack.utils.logging import logger

qc_name_dict = {
    'x2': 'ibmqx2',
    'melbourne': 'ibmq_16_melbourne',
    'athens': 'ibmq_athens',
    'santiago': 'ibmq_santiago',
    'lima': 'ibmq_lima',
    'belem': 'ibmq_belem',
    'quito': 'ibmq_quito',
}

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--n', type=int, default=4, help='number of qubit')
    parser.add_argument('--m', type=int, default=0, help='minutes of '
                                                         'reservation '
                                                         'lookahead')
    parser.add_argument('--name', type=str, default=None, help='qc name')

    args = parser.parse_args()

    IBMQ.load_account()
    provider = IBMQ.get_provider("ibm-q")

    if args.name is not None:
        circ = QuantumCircuit(1, 1)
        circ.h(0)
        circ.measure(0, 0)
        logger.info(f"Queue on {args.name}")
        backend = provider.get_backend(qc_name_dict[args.name])
        job1 = execute(circ, backend)
        job2 = execute(circ, backend)
        print('here1')
        job_monitor(job1, interval=1)
        print('here2')
        job_monitor(job2, interval=1)

        exit(0)

    backends = provider.backends(
                  filters=lambda x: x.configuration().n_qubits >= args.n and
                  not x.configuration().simulator and x.status().operational)

    candidates = []
    now = datetime.now()
    for back in backends:
        backend_status = back.status()
        if not backend_status.operational or \
                backend_status.status_msg != 'active':
            continue
        if isinstance(back, IBMQBackend):
            end_time = now + timedelta(minutes=args.m)
            try:
                if back.reservations(now, end_time):
                    logger.warning(
                        f"{back} jobs: {back.status().pending_jobs}, "
                        f"has reservation in {args.m} minutes.")
                    continue
                else:
                    logger.info(f"{back} jobs: {back.status().pending_jobs}, "
                                f"has no reservation in {args.m} minutes.")
            except Exception as err:  # pylint: disable=broad-except
                logger.warning("Unable to find backend reservation "
                               "information. "
                               "It will not be taken into consideration. %s",
                               str(err))
        candidates.append(back)
    if not candidates:
        raise ValueError('No backend matches the criteria.')

    least_busy_dev = min(candidates, key=lambda b: b.status().pending_jobs)

    logger.info(f"Least busy device: {least_busy_dev}")
