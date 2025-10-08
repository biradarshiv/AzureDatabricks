create table source_cars_data
(
Branch_ID varchar(200),
Dealer_ID varchar(200),
Model_ID varchar(200),
Revenue bigint,
Units_Sold bigint,
Date_ID  varchar(200),
Day int,
Month int,
Year int,
BranchName varchar(200),
DealerName varchar(200),
Product_Name varchar(200)
);


select * from source_cars_data;

create table water_table
(
last_load varchar(2000)
);



select min(Date_ID) from [dbo].[source_cars_data]; -- DT00001

truncate table water_table;

insert into water_table
values('DT00000');

select * from water_table;

select count(*) from [dbo].[source_cars_data] where Date_ID > 'DT00000'; --  1849
select count(*) from [dbo].[source_cars_data] ; --  1849



create or alter procedure UpdateWatermarkTable
@lastload varchar(200)
as
begin
begin transaction;
update water_table set last_load = @lastload;
commit transaction;
end;



