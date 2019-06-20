x_dimension = [9, 15, 25, 14, 10, 18, 0, 16, 5, 19, 16, 20]
y_dimension = [39, 56, 93, 61, 50, 75, 32, 85, 42, 70, 66, 80]

x_mean = sum(x_dimension)/len(x_dimension)
y_mean = sum(y_dimension)/len(y_dimension)

print(f"x mean = {x_mean}\ny mean = {y_mean}\n")

normal_x = [value - x_mean for value in x_dimension]
normal_y = [value - y_mean for value in y_dimension]
print()
print("with subtracted mean")
print()
print(f"normalized x  = {normal_x} \nnormalized y = {normal_y}")
print()
print()
print()
print(f"(H - Hbar)(M - Mbar) = ", [value1*value2 for (value1, value2) in zip(normal_x, normal_y)])
print()
print(f"sum of (H - Hbar)(M - Mbar) = ", sum([value1*value2 for (value1, value2) in zip(normal_x, normal_y)]))
print()


def covariance_matrix():
    global normal_y, normal_x
    dimensions = int(input("How many dimensions are there?"))
    print()
    co_var_matrix = [[None]*dimensions]*dimensions
    print(co_var_matrix)
    print()

    covar_x_x = sum([value**2 for value in normal_x])/len(x_dimension)-1
    covar_y_y = sum([value**2 for value in normal_y])/len(y_dimension)-1

    covar_x_y = sum([value1*value2 for (value1, value2) in zip(normal_x, normal_y)])/len(x_dimension)-1

    print(f"covar_x_x = {covar_x_x}")
    print()
    print(f"covar_x_y = {covar_x_y}")
    print()
    print(f"covar y_y = {covar_y_y}")
    print()
    i = 0
    c = covar_x_x
    for i_list in co_var_matrix:
        for index in range(len(i_list)):
            if i == index:
                co_var_matrix[i][index] = c
                c = covar_y_y
            else:
                co_var_matrix[i][index] = covar_x_y

        i = i+1

    for listo in co_var_matrix:
        print(listo)



covariance_matrix()