# """set1"""
# # importing the multiprocessing module 
# import multiprocessing 
  
# def print_cube(num): 
#     """ 
#     function to print cube of given num 
#     """
#     print("Cube: {}".format(num * num * num)) 
  
# def print_square(num): 
#     """ 
#     function to print square of given num 
#     """
#     print("Square: {}".format(num * num)) 
  
# if __name__ == "__main__": 
#     # creating processes 
#     p1 = multiprocessing.Process(target=print_square, args=(10, )) 
#     p2 = multiprocessing.Process(target=print_cube, args=(10, )) 
  
#     # starting process 1 
#     p1.start() 
#     # starting process 2 
#     p2.start() 
  
#     # wait until process 1 is finished 
#     p1.join() 
#     # wait until process 2 is finished 
#     p2.join() 

#     # both processes finished 
#     print("Done!") 



# # importing the multiprocessing module 
# import multiprocessing 
# import os 
  
# def worker1(): 
#     # printing process id 
#     print("ID of process running worker1: {}".format(os.getpid())) 
  
# def worker2(): 
#     # printing process id 
#     print("ID of process running worker2: {}".format(os.getpid())) 
  
# if __name__ == "__main__": 
#     # printing main program process id 
#     print("ID of main process: {}".format(os.getpid())) 
  
#     # creating processes 
#     p1 = multiprocessing.Process(target=worker1) 
#     p2 = multiprocessing.Process(target=worker2) 
  
#     # starting processes 
#     p1.start() 
#     p2.start() 
  
#     # process IDs 
#     print("ID of process p1: {}".format(p1.pid)) 
#     print("ID of process p2: {}".format(p2.pid)) 
  
#     # wait until processes are finished 
#     p1.join() 
#     p2.join() 
  
#     # both processes finished 
#     print("Both processes finished execution!") 
  
#     # check if processes are alive 
#     print("Process p1 is alive: {}".format(p1.is_alive())) 
#     print("Process p2 is alive: {}".format(p2.is_alive())) 


# """set 2"""


# import multiprocessing 
  
# # empty list with global scope 
# result = [] 
  
# def square_list(mylist): 
#     """ 
#     function to square a given list 
#     """
#     global result 
#     # append squares of mylist to global list result 
#     for num in mylist: 
#         result.append(num * num) 
#     # print global list result 
#     print("Result(in process p1): {}".format(result)) 
  
# if __name__ == "__main__": 
#     # input list 
#     mylist = [1,2,3,4] 
  
#     # creating new process 
#     p1 = multiprocessing.Process(target=square_list, args=(mylist,)) 
#     # starting process 
#     p1.start() 
#     # wait until process is finished 
#     p1.join() 
  
#     # print global result list 
#     print("Result(in main program): {}".format(result)) 




# import multiprocessing 
  
# def square_list(mylist, result, square_sum): 
#     """ 
#     function to square a given list 
#     """
#     # append squares of mylist to result array 
#     for idx, num in enumerate(mylist): 
#         result[idx] = num * num 
  
#     # square_sum value 
#     square_sum.value = sum(result) 
  
#     # print result Array 
#     print("Result(in process p1): {}".format(result[:])) 
  
#     # print square_sum Value 
#     print("Sum of squares(in process p1): {}".format(square_sum.value)) 
  
# if __name__ == "__main__": 
#     # input list 
#     mylist = [1,2,3,4] 
  
#     # creating Array of int data type with space for 4 integers 
#     result = multiprocessing.Array('i', 4) 
  
#     # creating Value of int data type 
#     square_sum = multiprocessing.Value('i') 
  
#     # creating new process 
#     p1 = multiprocessing.Process(target=square_list, args=(mylist, result, square_sum)) 
  
#     # starting process 
#     p1.start() 
  
#     # wait until process is finished 
#     p1.join() 
  
#     # print result array 
#     print("Result(in main program): {}".format(result[:])) 
  
#     # print square_sum Value 
#     print("Sum of squares(in main program): {}".format(square_sum.value)) 


# import multiprocessing 
  
# def print_records(records): 
#     """ 
#     function to print record(tuples) in records(list) 
#     """
#     for record in records: 
#         print("Name: {0}\nScore: {1}\n".format(record[0], record[1])) 
  
# def insert_record(record, records): 
#     """ 
#     function to add a new record to records(list) 
#     """
#     records.append(record) 
#     print("New record added!\n") 
  
# if __name__ == '__main__': 
#     with multiprocessing.Manager() as manager: 
#         # creating a list in server process memory 
#         records = manager.list([('Sam', 10), ('Adam', 9), ('Kevin',9)]) 
#         # new record to be inserted in records 
#         new_record = ('Jeff', 8) 
  
#         # creating new processes 
#         p1 = multiprocessing.Process(target=insert_record, args=(new_record, records)) 
#         p2 = multiprocessing.Process(target=print_records, args=(records,)) 
  
#         # running process p1 to insert new record 
#         p1.start() 
#         p1.join() 
  
#         # running process p2 to print records 
#         p2.start() 
#         p2.join() 



# import multiprocessing 
  
# def square_list(mylist, q): 
#     """ 
#     function to square a given list 
#     """
#     # append squares of mylist to queue 
#     for num in mylist: 
#         q.put(num * num) 
#     print("queue is done")
  
# def print_queue(q): 
#     """ 
#     function to print queue elements 
#     """
#     print("Queue elements:") 
#     while not q.empty(): 
#         print(q.get()) 
#     print("Queue is now empty!") 
  
# if __name__ == "__main__": 
#     # input list 
#     mylist = [1,2,3,4] 
  
#     # creating multiprocessing Queue 
#     q = multiprocessing.Queue() 
  
#     # creating new processes 
#     p1 = multiprocessing.Process(target=square_list, args=(mylist, q)) 
#     p2 = multiprocessing.Process(target=print_queue, args=(q,)) 
  
#     # running process p1 to square list 
#     p1.start() 
#     p1.join() 
#     print('sssss')
#     # running process p2 to get queue elements 
#     p2.start() 
#     p2.join() 



# import multiprocessing 
  
# def sender(conn, msgs): 
#     """ 
#     function to send messages to other end of pipe 
#     """
#     for msg in msgs: 
#         conn.send(msg) 
#         print("Sent the message: {}".format(msg)) 
#     conn.close() 
  
# def receiver(conn): 
#     """ 
#     function to print the messages received from other 
#     end of pipe 
#     """
#     while 1: 
#         msg = conn.recv() 
#         if msg == "END": 
#             break
#         print("Received the message: {}".format(msg)) 
  
# if __name__ == "__main__": 
#     # messages to be sent 
#     msgs = ["hello", "hey", "hru?", "END"] 
  
#     # creating a pipe 
#     parent_conn, child_conn = multiprocessing.Pipe() 
  
#     # creating new processes 
#     p1 = multiprocessing.Process(target=sender, args=(parent_conn,msgs)) 
#     p2 = multiprocessing.Process(target=receiver, args=(child_conn,)) 
  
#     # running processes 
#     p1.start() 
#     p2.start() 
  
#     # wait until processes finish 
#     p1.join() 
#     p2.join() 



# Python program to illustrate  
# the concept of race condition 
# in multiprocessing 
# import multiprocessing 
  
# function to withdraw from account 
# def withdraw(balance):     
#     for _ in range(10000): 
#         balance.value = balance.value - 1
  
# # function to deposit to account 
# def deposit(balance):     
#     for _ in range(10000): 
#         balance.value = balance.value + 1
  
# def perform_transactions(): 
  
#     # initial balance (in shared memory) 
#     balance = multiprocessing.Value('i', 100) 
  
#     # creating new processes 
#     p1 = multiprocessing.Process(target=withdraw, args=(balance,)) 
#     p2 = multiprocessing.Process(target=deposit, args=(balance,)) 
  
#     # starting processes 
#     p1.start() 
#     p2.start() 
  
#     # wait until processes are finished 
#     p1.join() #stop current process until p1 finished
#     p2.join() 
  
#     # print final balance 
#     print("Final balance = {}".format(balance.value)) 
  
# if __name__ == "__main__": 
#     for _ in range(1): 
  
#         # perform same transaction process 10 times 
#         perform_transactions() 



# Python program to illustrate  
# the concept of locks 
# in multiprocessing 
# import multiprocessing 
  
# # function to withdraw from account 
# def withdraw(balance, lock):     
#     for _ in range(1000): 
#         lock.acquire()
#         print('wwww')
#         balance.value = balance.value - 1
#         lock.release() 
  
# # function to deposit to account 
# def deposit(balance, lock):     
#     for _ in range(100): 
#         lock.acquire() 
#         print('ddddd')
#         balance.value = balance.value + 1
#         lock.release() 
  
# def perform_transactions(): 
  
#     # initial balance (in shared memory) 
#     balance = multiprocessing.Value('i', 100) 
  
#     # creating a lock object 
#     lock = multiprocessing.Lock() 
  
#     # creating new processes 
#     p1 = multiprocessing.Process(target=withdraw, args=(balance,lock)) 
#     p2 = multiprocessing.Process(target=deposit, args=(balance,lock)) 
  
#     # starting processes 
#     p1.start() 
#     p2.start() 
  
#     # wait until processes are finished 
#     p1.join() 
#     p2.join() 
  
#     # print final balance 
#     print("Final balance = {}".format(balance.value)) 
  
# if __name__ == "__main__": 
#     for _ in range(1): 
  
#         # perform same transaction process 10 times 
#         perform_transactions() 

# import multiprocessing 
# import os 
  
# def square(n): 
#     print("Worker process id for {0}: {1}".format(n, os.getpid())) 
#     return (n*n) 
  
# if __name__ == "__main__": 
#     # input list 
#     mylist = {1,2,3,4,5} 
  
#     # creating a pool object 
#     p = multiprocessing.Pool() 
  
#     # map list to target function 
#     result = p.map(square, mylist) 
#     print(type(result))
#     print(result) 


result = []

def add():
    global result
    result.append(1)
    print(result)

add()
print(result)




# result = [] 
  
# def square_list(mylist): 
#     """ 
#     function to square a given list 
#     """
#     global result 
#     # append squares of mylist to global list result 
#     for num in mylist: 
#         result.append(num * num) 
#     # print global list result 
#     print("Result(in process p1): {}".format(result)) 

# square_list([1,2])
# print(result)