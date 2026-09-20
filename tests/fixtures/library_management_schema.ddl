-- Create the 'Authors' table
CREATE TABLE Authors (
    author_id INT PRIMARY KEY AUTO_INCREMENT,
    first_name VARCHAR(100) NOT NULL,
    last_name VARCHAR(100) NOT NULL,
    birth_date DATE,
    death_date DATE,
    nationality VARCHAR(100),
    biography TEXT
);

-- Create the 'Publishers' table
CREATE TABLE Publishers (
    publisher_id INT PRIMARY KEY AUTO_INCREMENT,
    name VARCHAR(255) NOT NULL,
    address VARCHAR(255),
    city VARCHAR(100),
    state VARCHAR(50),
    zip_code VARCHAR(10),
    country VARCHAR(100),
    contact_email VARCHAR(255),
    contact_phone VARCHAR(20)
);

-- Create the 'Books' table
CREATE TABLE Books (
    book_id INT PRIMARY KEY AUTO_INCREMENT,
    title VARCHAR(255) NOT NULL,
    author_id INT NOT NULL,
    publisher_id INT NOT NULL,
    isbn VARCHAR(20) UNIQUE NOT NULL,
    publication_date DATE,
    genre VARCHAR(100),
    edition VARCHAR(50),
    number_of_pages INT,
    language VARCHAR(50),
    format ENUM('Hardcover', 'Paperback', 'E-book', 'Audiobook'),
    FOREIGN KEY (author_id) REFERENCES Authors(author_id),
    FOREIGN KEY (publisher_id) REFERENCES Publishers(publisher_id)
);

-- Create the 'Library_Branches' table
CREATE TABLE Library_Branches (
    branch_id INT PRIMARY KEY AUTO_INCREMENT,
    name VARCHAR(255) NOT NULL,
    address VARCHAR(255) NOT NULL,
    city VARCHAR(100) NOT NULL,
    state VARCHAR(50) NOT NULL,
    zip_code VARCHAR(10),
    phone_number VARCHAR(20),
    manager_id INT, --  FK to Employees
    opening_hours VARCHAR(255)
);

-- Create the 'Library_Members' table
CREATE TABLE Library_Members (
    member_id INT PRIMARY KEY AUTO_INCREMENT,
    first_name VARCHAR(100) NOT NULL,
    last_name VARCHAR(100) NOT NULL,
    date_of_birth DATE,
    gender ENUM('Male', 'Female', 'Other'),
    address VARCHAR(255),
    city VARCHAR(100),
    state VARCHAR(50),
    zip_code VARCHAR(10),
    email VARCHAR(255) UNIQUE NOT NULL,
    phone_number VARCHAR(20),
    join_date DATE NOT NULL,
    membership_type ENUM('Regular', 'Premium', 'Student', 'Senior'),
    account_status ENUM('Active', 'Inactive', 'Suspended', 'Closed') DEFAULT 'Active'
);

-- Create the 'Book_Inventory' table
CREATE TABLE Book_Inventory (
    inventory_id INT PRIMARY KEY AUTO_INCREMENT,
    book_id INT NOT NULL,
    branch_id INT NOT NULL,
    quantity INT NOT NULL,
    available_quantity INT NOT NULL,
    condition ENUM('New', 'Good', 'Fair', 'Poor'),
    FOREIGN KEY (book_id) REFERENCES Books(book_id),
    FOREIGN KEY (branch_id) REFERENCES Library_Branches(branch_id)
);

-- Create the 'Book_Loans' table
CREATE TABLE Book_Loans (
    loan_id INT PRIMARY KEY AUTO_INCREMENT,
    member_id INT NOT NULL,
    inventory_id INT NOT NULL,
    loan_date DATETIME NOT NULL,
    due_date DATE NOT NULL,
    return_date DATETIME,
    loan_status ENUM('Checked Out', 'Returned', 'Overdue', 'Lost', 'Renewed') DEFAULT 'Checked Out',
    fine_amount DECIMAL(10, 2) DEFAULT 0,
    FOREIGN KEY (member_id) REFERENCES Library_Members(member_id),
    FOREIGN KEY (inventory_id) REFERENCES Book_Inventory(inventory_id)
);
-- Create the 'Employees' table
CREATE TABLE Employees (
    employee_id INT PRIMARY KEY AUTO_INCREMENT,
    first_name VARCHAR(100) NOT NULL,
    last_name VARCHAR(100) NOT NULL,
    job_title VARCHAR(100),
    department_id INT,  -- FK to Departments
    branch_id INT,      -- FK to Library_Branches
    email VARCHAR(255),
    phone_number VARCHAR(20),
    hire_date DATE,
    FOREIGN KEY (department_id) REFERENCES Departments(department_id),
    FOREIGN KEY (branch_id) REFERENCES Library_Branches(branch_id)
);

-- Create the 'Departments' table
CREATE TABLE Departments (
    department_id INT PRIMARY KEY AUTO_INCREMENT,
    name VARCHAR(100),
    location VARCHAR(255),
    manager_id INT, -- FK to Employees
    FOREIGN KEY (manager_id) REFERENCES Employees(employee_id)
);
ALTER TABLE Library_Branches
ADD CONSTRAINT FK_Library_Branches_ManagerID FOREIGN KEY (manager_id) REFERENCES Employees(employee_id);
