# Applied Database Technologies — Fundamentals

## What is a Database?

A **database** is an organized collection of structured data stored and accessed electronically. A **Database Management System (DBMS)** is the software that manages databases, providing tools to create, read, update, and delete data.

A **Relational Database Management System (RDBMS)** stores data in tables (relations) and uses SQL (Structured Query Language) to interact with that data. Examples: PostgreSQL, MySQL, SQLite, Oracle, SQL Server.

## The Relational Model

In the relational model, data is organized into:

- **Tables (Relations)**: A collection of rows and columns. Each table represents an entity type (e.g., Customers, Orders).
- **Rows (Tuples)**: Individual records in a table.
- **Columns (Attributes)**: The fields that describe each record.
- **Primary Key**: A column or combination of columns that uniquely identifies each row. Cannot be NULL.
- **Foreign Key**: A column in one table that references the primary key of another table. Enforces referential integrity.

## Database Normalization

**Normalization** is the process of organizing a database schema to reduce data redundancy and improve data integrity. It involves decomposing tables into smaller, well-structured tables and defining relationships between them.

**Goals of normalization:**
1. Eliminate redundant (duplicate) data storage
2. Ensure data dependencies make sense (only related data is stored in a table)
3. Prevent update, insertion, and deletion anomalies

### Normal Forms

#### First Normal Form (1NF)
A table is in 1NF if:
- All columns contain atomic (indivisible) values — no sets or arrays in a single cell
- Each row is unique (has a primary key)
- All entries in a column are of the same data type

**Violation example:** A "PhoneNumbers" column containing "555-1234, 555-5678" violates 1NF because it contains multiple values.

**Fix:** Create a separate table for phone numbers with a foreign key back to the original table.

#### Second Normal Form (2NF)
A table is in 2NF if:
- It is in 1NF
- Every non-key attribute is **fully functionally dependent** on the entire primary key (no partial dependencies)

2NF only applies to tables with **composite primary keys** (keys made of multiple columns).

**Violation example:** A table with composite key (StudentID, CourseID) and columns StudentName, CourseName, Grade:
- StudentName depends only on StudentID (partial dependency)
- CourseName depends only on CourseID (partial dependency)
- Grade depends on both StudentID and CourseID (full dependency ✓)

**Fix:** Move StudentName to a Students table, CourseName to a Courses table.

#### Third Normal Form (3NF)
A table is in 3NF if:
- It is in 2NF
- No **transitive dependencies** exist — non-key attributes do not depend on other non-key attributes

**Violation example:** A table with columns: EmployeeID (PK), DepartmentID, DepartmentName
- DepartmentName depends on DepartmentID, which is not the primary key — transitive dependency

**Fix:** Move DepartmentID and DepartmentName to a separate Departments table.

#### Boyce-Codd Normal Form (BCNF)
A stronger version of 3NF. A table is in BCNF if:
- For every functional dependency X → Y, X must be a superkey (can uniquely identify any row)

BCNF eliminates all anomalies from functional dependencies. Most practical designs aim for 3NF or BCNF.

### Denormalization
Intentionally introducing redundancy for performance reasons (e.g., in data warehouses or read-heavy systems). Reduces costly JOINs at the expense of storage and update complexity.

## SQL — Structured Query Language

SQL is the standard language for interacting with relational databases.

### Data Definition Language (DDL)
```sql
CREATE TABLE Students (
    student_id INT PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    email VARCHAR(100) UNIQUE,
    enrollment_date DATE
);

ALTER TABLE Students ADD COLUMN major VARCHAR(50);
DROP TABLE Students;
```

### Data Manipulation Language (DML)
```sql
-- Insert
INSERT INTO Students (student_id, name, email) VALUES (1, 'Alice', 'alice@uni.edu');

-- Select
SELECT name, email FROM Students WHERE enrollment_date > '2023-01-01';

-- Update
UPDATE Students SET major = 'Computer Science' WHERE student_id = 1;

-- Delete
DELETE FROM Students WHERE student_id = 1;
```

### Joins
JOINs combine rows from two or more tables based on a related column.

```sql
-- INNER JOIN: only rows with matching values in both tables
SELECT s.name, e.course_id
FROM Students s
INNER JOIN Enrollments e ON s.student_id = e.student_id;

-- LEFT JOIN: all rows from the left table, matched rows from the right
SELECT s.name, e.course_id
FROM Students s
LEFT JOIN Enrollments e ON s.student_id = e.student_id;

-- RIGHT JOIN: all rows from the right table
-- FULL OUTER JOIN: all rows from both tables
```

### Aggregation
```sql
SELECT department, COUNT(*) AS num_students, AVG(gpa) AS avg_gpa
FROM Students
GROUP BY department
HAVING AVG(gpa) > 3.0
ORDER BY avg_gpa DESC;
```

### Subqueries
```sql
SELECT name FROM Students
WHERE student_id IN (
    SELECT student_id FROM Enrollments WHERE course_id = 'CS101'
);
```

## ACID Properties and Transactions

A **transaction** is a sequence of SQL operations that must be treated as a single unit of work. ACID properties guarantee reliable transaction processing.

### Atomicity
A transaction is all-or-nothing. Either all operations succeed, or the entire transaction is rolled back to its previous state. There are no partial transactions.

**Example:** Transferring money between accounts — both the debit and credit must succeed or both fail.

### Consistency
A transaction brings the database from one valid state to another. All data integrity constraints (primary keys, foreign keys, check constraints) must be satisfied before and after the transaction.

### Isolation
Concurrent transactions execute as if they were running in serial order. One transaction's intermediate state is not visible to other transactions.

**Isolation levels (weakest to strongest):**
- Read Uncommitted: can read uncommitted changes (dirty reads possible)
- Read Committed: can only read committed data
- Repeatable Read: same query returns same results within a transaction
- Serializable: full isolation, as if transactions ran one at a time

### Durability
Once a transaction is committed, it persists even in the event of a system failure (power outage, crash). Achieved through transaction logs and write-ahead logging (WAL).

## Indexes

An **index** is a data structure that improves the speed of data retrieval operations. Like a book index, it allows the database to find rows quickly without scanning the entire table (full table scan).

### B-Tree Index (Default)
Most common index type. Stored as a balanced tree that supports:
- Equality lookups: `WHERE id = 42`
- Range queries: `WHERE age BETWEEN 20 AND 30`
- Sorting: `ORDER BY name`

### Hash Index
Optimized for exact equality lookups only. Does not support range queries or sorting.

### Tradeoffs
- **Benefits**: Faster SELECT queries
- **Costs**: Slower INSERT/UPDATE/DELETE (index must be maintained), additional storage

**When to index:** Columns frequently used in WHERE, JOIN, or ORDER BY clauses, especially on large tables. Avoid indexing every column.

## NL2SQL (Natural Language to SQL)

NL2SQL (also called Text-to-SQL) is a technology that converts natural language questions into SQL queries, allowing users to query databases without knowing SQL syntax.

### How NL2SQL Works
1. **Schema understanding**: The system receives the database schema (table names, column names, types, relationships)
2. **Natural language parsing**: The system parses the user's question to understand intent, entities, and relationships
3. **SQL generation**: The system generates the corresponding SQL query
4. **Execution**: The SQL is executed against the database and results are returned

### Example
- **User question**: "How many students enrolled in Computer Science courses last year?"
- **Generated SQL**: 
```sql
SELECT COUNT(DISTINCT s.student_id)
FROM Students s
JOIN Enrollments e ON s.student_id = e.student_id
JOIN Courses c ON e.course_id = c.course_id
WHERE c.department = 'Computer Science'
AND YEAR(e.enrollment_date) = YEAR(CURRENT_DATE) - 1;
```

### NL2SQL Approaches
- **Rule-based**: Hand-crafted rules and templates (limited flexibility)
- **Machine learning**: Seq2seq models, BERT-based encoders
- **Large Language Models (LLMs)**: Modern approach using few-shot prompting. LLMs like GPT-4 and Claude can generate accurate SQL when given the schema as context.

### Challenges
- Handling ambiguous questions
- Multi-table joins and complex aggregations
- Domain-specific terminology
- Schema changes requiring model retraining (for non-LLM approaches)

### Using LLMs for NL2SQL
```python
# Example workflow
schema = """
Tables:
- students(student_id, name, email, major)
- courses(course_id, title, credits, instructor)
- enrollments(student_id, course_id, grade, semester)
"""

prompt = f"""
Given this database schema:
{schema}

Convert the following question to SQL:
"List all students who got an A in any course taught by Professor Smith"
"""
# Send to LLM → receive SQL query → execute against database
```

### Storage and Retrieval with NL2SQL
NL2SQL enables storing data through INSERT queries and retrieving data through SELECT queries, all generated from natural language. This dramatically lowers the barrier to using databases for non-technical users.

For data insertion: "Add a new student named Bob with email bob@uni.edu majoring in Physics"
→ `INSERT INTO students (name, email, major) VALUES ('Bob', 'bob@uni.edu', 'Physics');`

For retrieval: "Show me all Physics students sorted by name"
→ `SELECT * FROM students WHERE major = 'Physics' ORDER BY name;`
