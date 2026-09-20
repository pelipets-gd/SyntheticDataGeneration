-- Create the 'Companies' table
CREATE TABLE Companies (
    company_id INT PRIMARY KEY AUTO_INCREMENT,
    name VARCHAR(255) NOT NULL,
    address VARCHAR(255) NOT NULL,
    city VARCHAR(100) NOT NULL,
    state VARCHAR(50) NOT NULL,
    zip_code VARCHAR(10),
    phone_number VARCHAR(20),
    industry VARCHAR(100),
    website VARCHAR(255)
);

-- Create the 'Departments' table
CREATE TABLE Departments (
    department_id INT PRIMARY KEY AUTO_INCREMENT,
    company_id INT NOT NULL,
    name VARCHAR(100) NOT NULL,
    location VARCHAR(255),
    manager_id INT,  -- Foreign key to Employees table (can be self-referencing)
    FOREIGN KEY (company_id) REFERENCES Companies(company_id)
);

-- Create the 'Employees' table
CREATE TABLE Employees (
    employee_id INT PRIMARY KEY AUTO_INCREMENT,
    first_name VARCHAR(100) NOT NULL,
    last_name VARCHAR(100) NOT NULL,
    middle_name VARCHAR(100),
    email VARCHAR(255) UNIQUE NOT NULL,
    phone_number VARCHAR(20),
    hire_date DATE NOT NULL,
    job_title VARCHAR(100) NOT NULL,
    department_id INT NOT NULL,
    salary DECIMAL(10, 2) NOT NULL,
    employment_status ENUM('Full-time', 'Part-time', 'Contract', 'Temporary') NOT NULL,
    FOREIGN KEY (department_id) REFERENCES Departments(department_id)
);

-- Create the 'Projects' table
CREATE TABLE Projects (
    project_id INT PRIMARY KEY AUTO_INCREMENT,
    name VARCHAR(255) NOT NULL,
    description TEXT,
    start_date DATE NOT NULL,
    end_date DATE,
    company_id INT NOT NULL,
    project_status ENUM('Planning', 'In Progress', 'Completed', 'On Hold', 'Cancelled') NOT NULL,
    budget DECIMAL(12, 2),
    FOREIGN KEY (company_id) REFERENCES Companies(company_id)
);

-- Create the 'Employee_Projects' table (Junction table for Employees and Projects)
CREATE TABLE Employee_Projects (
    employee_project_id INT PRIMARY KEY AUTO_INCREMENT,
    employee_id INT NOT NULL,
    project_id INT NOT NULL,
    role VARCHAR(100),
    hours_worked DECIMAL(8, 2),
    start_date DATE,
    end_date DATE,
    FOREIGN KEY (employee_id) REFERENCES Employees(employee_id),
    FOREIGN KEY (project_id) REFERENCES Projects(project_id)
);

-- Create the 'Employee_Benefits' table
CREATE TABLE Employee_Benefits (
    benefit_id INT PRIMARY KEY AUTO_INCREMENT,
    employee_id INT NOT NULL,
    benefit_type VARCHAR(100) NOT NULL,
    enrollment_date DATE NOT NULL,
    coverage_amount DECIMAL(10, 2),
    FOREIGN KEY (employee_id) REFERENCES Employees(employee_id)
);

-- Create the 'Performance_Reviews' table
CREATE TABLE Performance_Reviews (
    review_id INT PRIMARY KEY AUTO_INCREMENT,
    employee_id INT NOT NULL,
    reviewer_id INT NOT NULL,  -- Self-referencing foreign key
    review_date DATE NOT NULL,
    rating INT NOT NULL CHECK (rating >= 1 AND rating <= 5),
    comments TEXT,
    goals_set TEXT,
    review_status ENUM('Draft', 'Final', 'Approved') DEFAULT 'Draft',
    FOREIGN KEY (employee_id) REFERENCES Employees(employee_id),
    FOREIGN KEY (reviewer_id) REFERENCES Employees(employee_id)
);
