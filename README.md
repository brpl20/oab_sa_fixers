# OAB Fixers
- Auxiliadores para arrumar o banco de dados de scraping em geral

# States
States:
Lawyer.where(state: [nil, ""]).count
    1527
Lawyer.where("LENGTH(TRIM(state)) = 1").count
    262
Lawyer.where("LENGTH(TRIM(state)) > 2").count
    72618

# Principal Lawyer
Lawyer.update_all(principal_lawyer_id: nil)

## Trim all states to first 2 characters and convert to uppercase
Lawyer.where.not(state: [nil, ""]).update_all("state = UPPER(LEFT(TRIM(state), 2))")
Lawyer.where("LENGTH(TRIM(state)) > 2").count
    0
Lawyer.where("LENGTH(TRIM(state)) = 1").count
    288
Lawyer.where(state: [nil, ""]).count
    1527

# Phones
Phone numers: it must contains only: numers, ( ) and - ... like in the exmale i i show you : (31)3298-5600 --- this one i need bettter inspectcion so we going to list everything before ..

Phone numers: it must contains only: numers, ( ) and - ... like in the exmale i i show you : (31)3298-5600 --- this one i need bettter inspectcion so we going to list everything before ..


# CEP/ZIP
Lawyer.where("zip_code ~ '[^0-9-]'").count
19814

Primeira Correcao:
Lawyer Count (414.1ms)  SELECT COUNT(*) FROM "lawyers" WHERE (zip_code ~ '[^0-9-]') /*application='LegalDataApi'*/
18807

legal-data-api(dev)> Lawyer.where("zip_code ~ '[a-zA-Z]'").count
  Lawyer Count (230.8ms)  SELECT COUNT(*) FROM "lawyers" WHERE (zip_code ~ '[a-zA-Z]') /*application='LegalDataApi'*/
=> 17947
