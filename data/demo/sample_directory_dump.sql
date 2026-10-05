-- Sample SQL Directory Database Dump
CREATE TABLE IF NOT EXISTS member_contacts (
    contact_id INTEGER PRIMARY KEY,
    full_name VARCHAR(100),
    email_address VARCHAR(100),
    contact_no VARCHAR(50),
    residence_city VARCHAR(100)
);

INSERT INTO member_contacts (contact_id, full_name, email_address, contact_no, residence_city) VALUES
(101, 'Aisha Al-Mansoor', 'aisha.m@gulftech.ae', '+971 50 123 4567', 'Dubai'),
(102, 'David Kim', 'david.kim@fintech.co', '9776655443', 'Seoul'),
(103, 'Elena Rostova', 'elena.rostova@nexus.ai', '9988776655', 'Pune'),
(104, 'Lucas Silva', 'lucas.silva@rio.br', '+55 21 98765-4321', 'Rio de Janeiro');
