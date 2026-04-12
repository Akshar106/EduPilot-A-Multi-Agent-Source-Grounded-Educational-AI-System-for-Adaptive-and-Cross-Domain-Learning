# Applied Database Technologies — Advanced Topics

## NoSQL Databases

NoSQL ("Not only SQL") databases are non-relational databases designed for specific data models and flexible schemas. They trade strict ACID guarantees for scalability and performance.

### Types of NoSQL Databases

#### Key-Value Stores
Simplest NoSQL model. Each value is identified by a unique key. Extremely fast for lookups by key.
- **Use cases**: Caching, session management, real-time leaderboards
- **Examples**: Redis, DynamoDB (simple key-value mode), Memcached

#### Document Databases
Store semi-structured data as documents (typically JSON or BSON). Documents within a collection can have different fields.
- **Use cases**: Content management, user profiles, catalogs
- **Examples**: MongoDB, CouchDB, Firestore

```json
{
  "student_id": "s001",
  "name": "Alice",
  "courses": ["CS101", "MATH201"],
  "address": {
    "city": "Boston",
    "state": "MA"
  }
}
```

#### Column-Family Stores
Store data in column families rather than rows. Optimized for queries over large datasets.
- **Use cases**: Time-series data, IoT, analytics at scale
- **Examples**: Apache Cassandra, HBase

#### Graph Databases
Represent data as nodes (entities) and edges (relationships). Optimized for traversing relationships.
- **Use cases**: Social networks, recommendation engines, fraud detection, knowledge graphs
- **Examples**: Neo4j, Amazon Neptune

### SQL vs. NoSQL Comparison

| Feature | SQL (Relational) | NoSQL |
|---------|-----------------|-------|
| Schema | Fixed, predefined | Flexible, dynamic |
| Scalability | Vertical (scale up) | Horizontal (scale out) |
| ACID | Full support | Varies (eventual consistency common) |
| Joins | Native, efficient | Manual or limited |
| Query language | Standard SQL | Database-specific |
| Best for | Complex queries, transactions | High volume, flexible data |

### CAP Theorem
In distributed systems, you can guarantee at most two of three properties:
- **Consistency**: All nodes see the same data at the same time
- **Availability**: Every request receives a response
- **Partition Tolerance**: System continues operating despite network partitions

SQL databases prioritize Consistency + Partition Tolerance (CP).
Many NoSQL databases prioritize Availability + Partition Tolerance (AP) with eventual consistency.

## Query Optimization

The query optimizer is a core component of a DBMS that determines the most efficient execution plan for a SQL query.

### EXPLAIN and Query Plans
```sql
EXPLAIN SELECT s.name, e.grade
FROM students s
JOIN enrollments e ON s.student_id = e.student_id
WHERE s.major = 'Computer Science';
```

The EXPLAIN output shows whether the database uses an index scan or full table scan, and the estimated cost of each step.

### Common Optimization Techniques

1. **Use indexes** on frequently queried columns
2. **Select only needed columns** (avoid `SELECT *`)
3. **Filter early**: Apply WHERE conditions to reduce data early
4. **Avoid functions on indexed columns**: `WHERE YEAR(date) = 2023` prevents index use; use `WHERE date BETWEEN '2023-01-01' AND '2023-12-31'` instead
5. **Analyze join order**: Start with the smallest filtered result set
6. **Use LIMIT** to avoid returning more rows than needed
7. **Normalize appropriately**: Avoid storing derived values that can be computed

### Statistics and Cardinality
Query optimizers use statistics (number of rows, distribution of values) to estimate the cost of different execution plans. Run `ANALYZE` (PostgreSQL) or `UPDATE STATISTICS` (SQL Server) to keep statistics current.

## Stored Procedures and Triggers

### Stored Procedures
Pre-compiled SQL code stored in the database. Can accept parameters and return results.

```sql
CREATE PROCEDURE EnrollStudent(
    IN p_student_id INT,
    IN p_course_id VARCHAR(10),
    IN p_semester VARCHAR(10)
)
BEGIN
    -- Check if student exists
    IF NOT EXISTS (SELECT 1 FROM students WHERE student_id = p_student_id) THEN
        SIGNAL SQLSTATE '45000' SET MESSAGE_TEXT = 'Student not found';
    END IF;
    
    -- Insert enrollment
    INSERT INTO enrollments (student_id, course_id, semester)
    VALUES (p_student_id, p_course_id, p_semester);
END;
```

**Benefits:** Code reuse, reduced network traffic, security (users only need EXECUTE permission), performance (pre-compiled).

### Triggers
Automatically executed SQL code in response to INSERT, UPDATE, or DELETE on a table.

```sql
CREATE TRIGGER UpdateGPA
AFTER INSERT ON enrollments
FOR EACH ROW
BEGIN
    UPDATE students
    SET gpa = (
        SELECT AVG(grade_points)
        FROM enrollments
        WHERE student_id = NEW.student_id
    )
    WHERE student_id = NEW.student_id;
END;
```

**Use cases:** Audit trails, maintaining derived values, enforcing complex constraints.

## Entity-Relationship (ER) Modeling

An ER diagram is a conceptual data model used to design a database schema before implementation.

### Key Concepts
- **Entity**: A real-world object or concept (e.g., Student, Course)
- **Attribute**: A property of an entity (e.g., student_id, name)
- **Relationship**: An association between entities (e.g., Student ENROLLS IN Course)
- **Cardinality**: One-to-one (1:1), one-to-many (1:N), many-to-many (M:N)

### Converting ER to Relational Schema
- **Entities** → Tables
- **Attributes** → Columns
- **1:N relationships** → Foreign key in the "many" table
- **M:N relationships** → Junction table with foreign keys to both entities
- **1:1 relationships** → Foreign key in either table (or merge into one table)

## Database Concurrency Control

### Locking Mechanisms
- **Shared (Read) Lock**: Multiple transactions can read simultaneously; no writes allowed
- **Exclusive (Write) Lock**: Only one transaction can read or write; others must wait
- **Two-Phase Locking (2PL)**: Growing phase (acquire locks) then shrinking phase (release locks); guarantees serializability

### Deadlocks
A deadlock occurs when two or more transactions are each waiting for the other to release a lock.

**Detection**: Wait-for graph — deadlock exists if there's a cycle.
**Prevention**: Timestamps (wound-wait or wait-die protocols).
**Resolution**: Abort one of the deadlocked transactions.

### Multiversion Concurrency Control (MVCC)
Modern databases (PostgreSQL, MySQL InnoDB) use MVCC to allow readers and writers to not block each other. Each write creates a new version of the row; readers see a snapshot from when their transaction started.

## Data Warehousing and OLAP

### OLTP vs. OLAP
- **OLTP (Online Transaction Processing)**: Many short transactions, normalized schema, optimized for insert/update/delete (operational databases)
- **OLAP (Online Analytical Processing)**: Few complex queries, denormalized star/snowflake schema, optimized for aggregations and reporting (data warehouses)

### Star Schema
A data warehouse schema with a central fact table (containing measurements/metrics) surrounded by dimension tables (containing descriptive attributes).

**Example:**
- Fact table: Sales (sale_id, date_id, product_id, customer_id, amount, quantity)
- Dimension tables: Date, Product, Customer

## Database Security

### Authentication and Authorization
- **Authentication**: Verifying who the user is (username/password, certificates)
- **Authorization**: Controlling what the user can do (GRANT/REVOKE privileges)

```sql
GRANT SELECT, INSERT ON students TO readonly_user;
REVOKE DELETE ON students FROM junior_staff;
```

### SQL Injection
A critical security vulnerability where malicious SQL code is injected into an application query through user input.

**Vulnerable code (Python):**
```python
query = f"SELECT * FROM users WHERE name = '{user_input}'"
# If user_input = "' OR '1'='1", the query returns all users!
```

**Prevention:**
- Use parameterized queries / prepared statements
- Validate and sanitize all user inputs
- Apply principle of least privilege

```python
# Safe parameterized query
cursor.execute("SELECT * FROM users WHERE name = %s", (user_input,))
```
